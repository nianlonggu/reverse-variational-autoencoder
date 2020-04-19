
import time
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import imageio
import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import glob
import sys




# def prepare_data(data_dir, is_train):
#     transform = transforms.Compose([transforms.ToTensor()])
#     mnist_train = datasets.MNIST( root="./data", train= is_train, download= True, transform= transform )
#     train_loader = DataLoader( mnist_train, batch_size= len(mnist_train) )
#     mnist_train_array = next( iter(train_loader) )[0].numpy()
#     os.system("rm -rf "+data_dir+"image")
#     os.makedirs(data_dir+"image")
#     idx = 0
#     for img in mnist_train_array:
#         imageio.imsave( data_dir+"image/%05d.jpg"%(idx),  (img[0]*255).astype(np.uint8) )
#         idx +=1
#         if idx % 1000 ==999:
#             print("\rProgress %.2f%%"%(  idx/mnist_train_array.shape[0]*100 ), end="", flush= True)
#     print()


class CustomImageLoader:
    def __init__(self, data_dir, image_size, batch_size, num_output_channels,  num_workers = 16):
        transform_list = [ transforms.Resize( image_size, 0 ),  
                           #here we need to use the nearest neighbor resize, to be consistent with the F.interpolate function. 0 stands for PIL.Image.NEAREST
                           transforms.ToTensor(),
                           transforms.Normalize( (0.5,), (0.5,) ) ]
        if num_output_channels == 1:
            transform_list = [ transforms.Grayscale( num_output_channels= 1 ) ]+ transform_list
        
        self.dset = datasets.ImageFolder( root = data_dir, transform = transforms.Compose( transform_list ))
        self.dataloader = DataLoader( self.dset, batch_size= batch_size, shuffle= True, num_workers= num_workers , drop_last= True)
        def cycle(dataloader):
            while True:
                for x in dataloader:
                    yield x
        self.dataiter = iter(cycle( self.dataloader ))
    def get_next( self ):
        return next(self.dataiter ) 



def save_model(  module_dicts ,save_name , max_to_keep = 0, overwrite = True  ):
    folder_path = os.path.dirname( os.path.abspath( save_name )  )
    if not os.path.exists( folder_path  ):
        os.makedirs( folder_path )

    state_dicts = {}
    for key in module_dicts.keys():
        if isinstance( module_dicts[key], nn.DataParallel ):
            state_dicts[key] = module_dicts[key].module.state_dict()
        else:
            state_dicts[key] = module_dicts[key].state_dict()

    if os.path.exists( save_name ):
        if overwrite:
            os.remove( save_name )
            torch.save( state_dicts, save_name )
        else:
            print("Warning: checkpoint file already exists!")
            return
    else:
        torch.save( state_dicts, save_name )

    if max_to_keep > 0:
        pt_file_list = glob.glob(folder_path+"/*.pt")
        pt_file_list.sort( key= lambda x: os.path.getmtime(x) )
        for idx in range( len( pt_file_list ) - max_to_keep ):
            os.remove( pt_file_list[idx]  )

def load_model( module_dicts, ckpt_file_name = None ):
    if not os.path.exists(ckpt_file_name):
        print("Warning: ckpt file not exists! Model loading failed!")
        return
    ckpt = torch.load( ckpt_file_name )
    for key in module_dicts.keys():
        module_dicts[key].load_state_dict( ckpt[key] )
    print("Model successfully loaded!")



def update_moving_average( m_ema, m, decay=0.999 ):
    with torch.no_grad():
        param_dict_m_ema =  m_ema.module.parameters()  if isinstance(  m_ema, nn.DataParallel ) else m_ema.parameters() 
        param_dict_m =  m.module.parameters()  if isinstance( m , nn.DataParallel ) else  m.parameters() 
        for param_m_ema, param_m in zip( param_dict_m_ema, param_dict_m ):
            param_m_ema.copy_( decay * param_m_ema + (1-decay) *  param_m )

def compute_gradient_penalty( real_data, fake_data, discriminator, stage, alpha, LAMBDA, device ):
    alpha_interp = torch.rand(real_data.size(0), 1,1,1)
    alpha_interp = alpha_interp.expand(real_data.size())
    alpha_interp = alpha_interp.to( device)

    # print(real_data.shape, fake_data.shape, alpha_interp.shape)
    interpolates = alpha_interp * real_data + ((1 - alpha_interp) * fake_data)
    interpolates = interpolates.to( device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates, stage, alpha )

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device) ,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty
