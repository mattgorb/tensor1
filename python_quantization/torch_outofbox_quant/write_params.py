import struct
import numpy as np

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

import torch.quantization
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub

args = None

class Net(nn.Module):
    def __init__(self, q = False):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, args.hidden_size, )
        self.fc2 = nn.Linear(args.hidden_size, 10, )
        self.q=q
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.numpy=True

    def forward(self, x, ):
        x = self.quant(x)
        if self.numpy:
            #ptq
            '''y=np.matmul(x[0,:].int_repr().detach().numpy(),   self.fc1.weight().int_repr().t().detach().numpy())*self.fc1.weight().q_scale()
            y=y+self.fc1.bias().detach().numpy()
            print(y)
            y=np.round(y)
            y=np.maximum(0,y)
            print(y)
            y=np.matmul(y,   self.fc2.weight().int_repr().t().detach().numpy())*self.fc2.weight().q_scale()
            y=y+self.fc2.bias().detach().numpy()
            y=np.round(y)
            print(y)'''

            #qat
            y=torch.matmul( x.int_repr().float()*0.003919653594493866 ,self.fc1.weight().int_repr().t().float())

            print(self.fc1.weight().q_zero_point())
            print(self.fc1.weight().q_scale())
            print(self.fc1.zero_point)
            print(self.fc1.scale)
            sys.exit()
            y=y*self.fc1.weight().q_scale()
            #print(y)
            y=y+self.fc1.bias()
            #print(y)
            #sys.exit()
            y=F.relu(y)
            
            y=(y/self.fc1.scale)+self.fc1.zero_point
            y=torch.round(y)
            y=(y-self.fc1.zero_point)*self.fc1.scale

            y=torch.matmul(y,self.fc2.weight().int_repr().float().t())
            y=y*self.fc2.weight().q_scale()
            y=y+self.fc2.bias()

            y=(y/self.fc2.scale)+self.fc2.zero_point
            y=torch.round(y)
            y=(y-self.fc2.zero_point)*self.fc2.scale


 
  

        x = self.fc1(x)
        x = F.relu(x)        
        x = self.fc2(x)
        # print(x)
        x = self.dequant(x)
        if self.numpy:
            print(torch.sum((y-x).abs()))
        return x
    
def write_images(model,model_np, device, criterion, test_loader, best_acc, save=False):
    model.eval()

    imgs=[]
    with torch.no_grad():
        for i,(data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data.view(-1, 784))
            print(f"Image {i}")
            print(output)
            data_round=(data>0.5).view(-1, 784).to(torch.int8).cpu().detach().numpy()[0]
            imgs.append( ','.join([str(x) for x in data_round[:-1]]))

            if i>3:
                break
    print(len(imgs))    
    
    if save:
        images=[f'uint16_t image_rounded_{i}[]= {{{imgs[i]}}};' for i in range(len(imgs))]
        images_str = '\n'.join([x for x in images])
        with open('mnist_images_rounded.h', 'w') as cpp_file:
            cpp_file.write(images_str)

    imgs=[]
    with torch.no_grad():
        for i,(data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data.view(-1, 784))
            print(f"Image {i}")
            print(output)
            data_round=data.view(-1, 784).cpu().detach().numpy()[0]
            imgs.append( ','.join([str(x) for x in data_round[:-1]]))
            if i>3:
                break
    
    if save:
        images=[f'float image_{i}[]= {{{imgs[i]}}};' for i in range(len(imgs))]
        images_str = '\n'.join([x for x in images])
        with open('mnist_images.h', 'w') as cpp_file:
            cpp_file.write(images_str)

    imgs=[]
    with torch.no_grad():
        for i,(data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data.view(-1, 784), )

            print(f"Image {i}")
            print(output)
            #print(torch.round(data/model_np['quant_scale']))
            #print(torch.round(data/model_np['quant_scale']))
            image=torch.round(data/model_np['quant_scale']).to(torch.uint8)
            image=image.view(-1, 784).cpu().detach().numpy()[0]
            imgs.append( ','.join([str(x) for x in image[:-1]]))
            if i>3:
                break
    
    if save:
        images=[f'uint8_t image_{i}_int[]= {{{imgs[i]}}};' for i in range(len(imgs))]
        images_str = '\n'.join([x for x in images])
        with open('mnist_images_int.h', 'w') as cpp_file:
            cpp_file.write(images_str)

def write_c_blob(model_dict, filename):
    """
    Dumps parameter data

    Arguments:
        filename (string): filename for binary blob
    """

    #filename = filename + '.c'
    byte_array=[]

    name_ls=[]
    dtype_ls=[]
    dim_ls=[]
    shape_ls=[]
    byte_array=[]
    for w_name in model_dict.keys():

        w = model_dict[w_name]

        name_ls.append(w_name.replace('.','_'))
        dtype_ls.append(w.dtype)
        dim_ls.append(w.ndim)

        shape_ls.append(','.join([str(w.shape[i]) for i in range(w.ndim)]))
        byte_array.append(','.join(str(b) for b in w.tobytes()))


    dtype_names=[f'const char *{i}_dtype="{j}"; ' for i,j in zip(name_ls,dtype_ls)]
    dtype_names_str = '\n'.join([x for x in dtype_names])

    dim_names=[f'const char {i}_dim={j};' for i,j in zip(name_ls,dim_ls)]
    dim_names_str = '\n'.join([x for x in dim_names])

    shape_names=[f'const uint32_t {i}_shape[] = {{{j}}};' for i,j in zip(name_ls, shape_ls)]
    shape_names_str = '\n'.join([x for x in shape_names])

    #data_names=[f'const unsigned char {i}_data[]={{{j}}}; ' for i,j in zip(name_ls,byte_array)]
    data_names=[f'const uint8_t {i}_data[]={{{j}}}; ' for i,j in zip(name_ls,byte_array)]
    data_names_str = '\n'.join([x for x in data_names])

    size_names=[f'const uint64_t {i}_size = sizeof({i}_data);' for i,j in zip(name_ls, byte_array)]
    size_names_str = '\n'.join([x for x in size_names])

    print(dtype_names_str)
    print(dim_names_str)

    print(shape_names_str)
    #print(data_names_str)
    print(size_names_str)

    cpp_data=f"""
#include "{filename + '.h'}"
{dtype_names_str}
{dim_names_str}
{shape_names_str}
{data_names_str}
{size_names_str}
    """
    dtype_names=[f'extern const char *{i}_dtype; ' for i,j in zip(name_ls,dtype_ls)]
    dtype_names_str = '\n'.join([x for x in dtype_names])


    dim_names=[f'extern const char {i}_dim;' for i,j in zip(name_ls,dim_ls)]
    dim_names_str = '\n'.join([x for x in dim_names])

    
    shape_names=[f'extern const uint32_t {i}_shape[] ;' for i,j in zip(name_ls, shape_ls)]
    shape_names_str = '\n'.join([x for x in shape_names])

    size_names=[f'extern const uint64_t {i}_size ;' for i,j in zip(name_ls, data_names)]
    size_names_str = '\n'.join([x for x in size_names])

    
    #data_names=[f'extern const unsigned char {i}_data[]; ' for i,j in zip(name_ls,data_names)]
    data_names=[f'extern const uint8_t {i}_data[]; ' for i,j in zip(name_ls,data_names)]
    data_names_str = '\n'.join([x for x in data_names])


    with open(filename + '.c', 'w') as cpp_file:
        cpp_file.write(cpp_data)

    header_data=f"""
#include <stdint.h>
#include "tensor1.h"
#ifndef {filename.upper()}_H_
#define {filename.upper()}_H_
{dtype_names_str}
{dim_names_str}
{shape_names_str}
{size_names_str}
{data_names_str}
#endif  // {filename.upper()}_H_
    """

    with open( filename + '.h', 'w') as header_file:
        header_file.write(header_data)

    print(f"Length of CPP byte array: {len(byte_array)}")






def test(model, device, criterion, test_loader, best_acc):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data.view(-1, 784))
            test_loss += criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    acc=correct/len(test_loader.dataset)
    print("Test Accuracy: {}\n".format(acc))
    return acc





def main():
    global args
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--file_in', type=str, default=None)
    parser.add_argument('--file_out', type=str, default=None)
    parser.add_argument('--hidden_size', type=int, default=None)
    args = parser.parse_args()
    if args.file_in is None or args.file_out is None or args.hidden_size is None: 
        print("Set Arguments!")
        sys.exit(0)

    device = "cpu"



    model = Net().to(device)
    backend = "qnnpack"
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend
    #model = torch.quantization.prepare_qat(model)
    model=torch.quantization.prepare(model, inplace=False)
    model = torch.quantization.convert(model)
    model.load_state_dict(torch.load(f'../../weights/{args.file_in}.pt'))


    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(os.path.join('/s/luffy/b/nobackup/mgorb/data/', 'mnist'), train=False,
                        transform=transforms.Compose([ transforms.ToTensor(), ])),
        batch_size=1, shuffle=False,worker_init_fn=np.random.seed(0),)
    criterion = nn.CrossEntropyLoss().to(device)
    #test(model, device, criterion, test_loader, 0)


    model_np={}
    for name,param in model.state_dict().items():
        if args.file_in=='mnist_mlp_qat' and '_packed_params._packed_params' not in name and 'dtype' not in name:
            print(name)
            if 'quant' in name:
                model_np[name.replace('.', '_')]=param.detach().numpy()[0]
            else:
                model_np[name.replace('.', '_')]=param.detach().numpy()
        if '_packed_params._packed_params' in name:
                print(f"adding {name} to dictionary..")
                model_np[name.replace('_packed_params._packed_params','')+"weight"]=torch.int_repr(param[0]).cpu().detach().numpy()
                model_np[name.replace('_packed_params._packed_params','')+"bias"]=param[1].cpu().detach().numpy()
                model_np[name.replace('_packed_params._packed_params','')+"weight_scale"]=np.array(param[0].q_scale())
                model_np[name.replace('_packed_params._packed_params','')+"weight_zero_point"]=np.array(param[0].q_zero_point())

    

    print("")
    write_c_blob(model_np, f'{args.file_out}_{args.hidden_size}')


    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(os.path.join('/s/luffy/b/nobackup/mgorb/data/', 'mnist'), train=False,
                        transform=transforms.Compose([ transforms.ToTensor(), ])),
        batch_size=1, shuffle=False,worker_init_fn=np.random.seed(0),)
    write_images(model,model_np, device, None, test_loader, None, save=False)


if __name__ == '__main__':
    main()