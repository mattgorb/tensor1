from __future__ import print_function
import argparse
import os
import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.autograd as autograd
import numpy as np



class GetSubnetBinary(autograd.Function):
    @staticmethod
    def forward(ctx, scores, weights, k, alpha):
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())
        # flat_out and out access the same memory. switched 0 and 1
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        # Perform binary quantization of weights
        abs_wgt = torch.abs(weights.clone()) # Absolute value of original weights

        # Save absolute value of weights for backward
        ctx.save_for_backward(abs_wgt)

        # Return pruning mask with gain term alpha for binary weights
        return alpha * out

    @staticmethod
    def backward(ctx, g):
        # Get absolute value of weights from saved ctx
        abs_wgt, = ctx.saved_tensors
        # send the gradient g times abs_wgt on the backward pass
        return g * abs_wgt, None, None, None

class LinearSubnet(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        self.register_buffer('alpha' , torch.tensor(1, requires_grad=False))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False
        self.prune_rate = 0.5

    @property
    def clamped_scores(self):
        # For unquantized activations
        return self.scores.abs()



    '''def calc_alpha(self):
        abs_wgt = torch.abs(self.weight.clone()) # Absolute value of original weights
        q_weight = abs_wgt * self.scores.abs() # Remove pruned weights
        num_unpruned = int(self.prune_rate * self.scores.numel()) # Number of unpruned weights
        self.alpha = torch.sum(q_weight) / num_unpruned # Compute alpha = || q_weight ||_1 / (number of unpruned weights)
        return self.alpha'''

    def set_params(self):
        self.mask = torch.nn.Parameter(torch.sign(GetSubnetBinary.apply(self.clamped_scores, self.weight, self.prune_rate, self.calc_alpha())))
        self.weight=torch.nn.Parameter(torch.sign(self.weight))
        self.calc_alpha()

        del self.scores
        print(self.mask[0,:10])
        print(self.weight[0, :10])
        print(self.alpha)


    def forward(self, x):

        w = self.weight * self.mask * self.alpha
        # Pass binary subnetwork weights to convolution layer
        x= F.linear(
            x, w, self.bias
        )

        return x







class NetSparse(nn.Module):
    def __init__(self):
        super(NetSparse, self).__init__()
        self.fc1 = LinearSubnet(784, 256, bias=False)
        self.fc2 = LinearSubnet(256, 10, bias=False)


    def init(self):
        self.fc1.set_params()
        self.fc2.set_params()

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x








def test(model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data.view(-1, 784))
            test_loss += criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))




def predict_x_images(model, device, test_loader,num_images=1):
    for i in range(num_images):
        data, target = test_loader.dataset[i]
        model.eval()

        target=torch.tensor(target)

        data, target = data.to(device), target.to(device)
        output = model(data.view(-1, 784))
        pred = output.argmax(dim=1, keepdim=True)
        print(output)
        prob = output.max(dim=1, keepdim=True)
        print('save image')
        print(f'pred: {pred},prob: {prob} target: {target}')
        #save_image(data, f'images/img{i}.png')

        print(data.view(-1, 784))

def main():
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(os.path.join('/s/luffy/b/nobackup/mgorb/data/', 'mnist'), train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           #transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=64, shuffle=True,worker_init_fn=np.random.seed(0),)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss().to(device)


    model = NetSparse().to(device)
    model.load_state_dict(torch.load('../weights/mnist_mlp_biprop.pt'))
    model.init()

    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        if 'alpha' in param_tensor:
            print(model.state_dict()[param_tensor])


    test(model, device, criterion, test_loader,)
    predict_x_images(model, device, test_loader, num_images=1)
    torch.save(model.state_dict(), "../weights/mnist_mlp_biprop_reconfigured.pt")

if __name__ == '__main__':
    main()