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
import math
import numpy as np
np.set_printoptions(suppress=True)

from torchvision.utils import save_image

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

        self.mask = nn.Parameter(torch.Tensor(self.weight.size()))
        self.register_buffer('alpha' , torch.tensor(1, requires_grad=False,).to(torch.float32))

    def to_type(self, type):
        self.mask=nn.Parameter(self.mask.to(type), requires_grad=False)
        self.weight=nn.Parameter(self.weight.to(type), requires_grad=False)

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

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class NetSparsePacked(nn.Module):
    def __init__(self,):
        super(NetSparsePacked, self).__init__()
        def pack_col(num):
            return int(num/8) if num%8==0 else int(num/8+1)

        self.fc1 = LinearSubnet(pack_col(784), 256, bias=False)
        self.fc2 = LinearSubnet(pack_col(256), 10, bias=False)

        self.fc1.to_type(torch.uint8)
        self.fc2.to_type(torch.uint8)

    def unpack(self, packed_tensor,unpacked_size):
        matrix = packed_tensor.numpy()
        num_bits = 8  # Number of bits to extract

        shifted_matrix = matrix[:, :, np.newaxis] >> np.arange(num_bits - 1, -1, -1)
        extracted_bits_matrix = shifted_matrix & 1
        extracted_bits_matrix = extracted_bits_matrix.reshape(extracted_bits_matrix.shape[0], -1)[:, :unpacked_size]
        unpacked_binary_data = torch.tensor(extracted_bits_matrix)
        return unpacked_binary_data

    def forward(self, x):
        temp_unpacked=LinearSubnet(784, 256, bias=False)

        temp_unpacked.weight=nn.Parameter((2*self.unpack(self.fc1.weight,784)-1).to(torch.int8), requires_grad=False)
        temp_unpacked.mask=nn.Parameter((self.unpack(self.fc1.mask,784)).to(torch.int8), requires_grad=False)
        temp_unpacked.alpha = self.fc1.alpha



        x = temp_unpacked(x)
        del temp_unpacked

        x = F.relu(x)


        temp_unpacked=LinearSubnet(256, 10, bias=False)
        temp_unpacked.weight=nn.Parameter((2*self.unpack(self.fc2.weight,256)-1).to(torch.int8), requires_grad=False)
        temp_unpacked.mask=nn.Parameter((self.unpack(self.fc2.mask,256)).to(torch.int8), requires_grad=False)
        temp_unpacked.alpha = self.fc2.alpha
        x = temp_unpacked(x)
        del temp_unpacked

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




def save_x_images(model, device, test_loader,num_images=1):
    for i in range(num_images):
        data, target = test_loader.dataset[i]
        model.eval()

        target=torch.tensor(target)
        data, target = data.to(device), target.to(device)
        output = model(data.view(-1, 784))
        pred = output.argmax(dim=1, keepdim=True)
        print(output.cpu().detach().numpy())
        prob = output.max(dim=1, keepdim=True)
        print('save image')
        print(f'pred: {pred},prob: {prob} target: {target}')
        #save_image(data, f'../images/img{i}.png')

        print(data.view(-1, 784))

    #save

def pack(parameter):
    num_cols = parameter.size(1)
    pack_columns = int(parameter.size(1) / 8) if parameter.size(1) % 8 == 0 else math.floor(
        int(parameter.size(1) / 8) + 1)

    # Packing
    packed_values = []
    for row in parameter.to(torch.int8).numpy().tolist():
        row.extend([0] * (pack_columns * 8 - num_cols))
        result_list = [int(''.join(map(str, row[i:i + 8])), 2) for i in range(0, len(row), 8)]
        packed_values.append(result_list)

    packed_tensor = torch.tensor(packed_values, dtype=torch.uint8)
    return packed_tensor

def main():
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(os.path.join('/s/luffy/b/nobackup/mgorb/data/', 'mnist'), train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=64, shuffle=False,worker_init_fn=np.random.seed(0),)

    device = "cpu"
    criterion = nn.CrossEntropyLoss().to(device)


    model = NetSparse().to(device)
    model.load_state_dict(torch.load('../weights/mnist_mlp_biprop_reconfigured.pt'))

    packed_model = NetSparsePacked().to(device)

    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size(),"\t",model.state_dict()[param_tensor].dtype)

        if 'weight' in param_tensor:
            model.state_dict()[param_tensor].copy_(((model.state_dict()[param_tensor].to(torch.int8)+1)/2))
        if 'alpha' not in param_tensor:

            packed_tensor=pack(model.state_dict()[param_tensor])
            print(packed_tensor)
            
            packed_model.state_dict()[param_tensor].copy_(packed_tensor)

        else:
            packed_model.state_dict()[param_tensor].copy_(model.state_dict()[param_tensor])
        #print(model.state_dict()[param_tensor])


    print('\n\nPacked model info')
    for param_tensor in packed_model.state_dict():
        print(param_tensor, "\t", packed_model.state_dict()[param_tensor].size(),"\t",packed_model.state_dict()[param_tensor].dtype)
        if 'alpha' in param_tensor:
            print(packed_model.state_dict()[param_tensor])


    test(packed_model, device, criterion, test_loader,)
    save_x_images(packed_model, device, test_loader,num_images=1)

    torch.save(packed_model.state_dict(), "../weights/mnist_mlp_biprop_packed.pt")
    
if __name__ == '__main__':
    main()














