from utils import *

class PixelNorm( nn.Module):
    def __init__( self ):
        super( PixelNorm, self ).__init__()
        self.epsilon = 1e-8
    def forward( self, net ):
        return torch.div( net,  torch.sqrt( torch.mean( net**2, dim = 1, keepdim= True ) + self.epsilon ) )

class MinibatchStddev( nn.Module ):
    def __init__(self ):
        super( MinibatchStddev, self ).__init__()
        self.epsilon = 1e-8
    def forward( self, net ):
        tmp_net = net - torch.mean( net, dim = 0, keepdim = True )
        tmp_net = torch.mean( tmp_net.pow(2), dim = 0, keepdim = False )      
        torch_std = torch.sqrt( tmp_net + self.epsilon  )
        # better not create new tensor in forward function 
        # std_dev = torch.mean( torch_std ) * torch.ones( net.size(0), 1, net.size(2), net.size(3) ) # torch.ones will create new tensors in CPU by default
        std_dev = torch.mean( torch_std ).view(1,1,1,1).repeat( net.size(0), 1, net.size(2), net.size(3)   )
        net = torch.cat( [ net, std_dev ], dim= 1 )
        return net

class EqualizedConv2d( nn.Module ):
    def __init__( self,  in_channels, out_channels, kernel_size, stride, padding, bias = True  ):
        super( EqualizedConv2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.weight_param = nn.Parameter( torch.FloatTensor( out_channels, in_channels, kernel_size, kernel_size ).normal_(0.0, 1.0)  )
        self.bias_param = None
        if bias:
            self.bias_param = nn.Parameter( torch.FloatTensor( out_channels ).fill_(0)  )
        fan_in = kernel_size * kernel_size * in_channels
        self.scale = np.sqrt(  2./ fan_in )
    def forward( self, net ):
        return F.conv2d( input= net,
                         weight = torch.mul( self.weight_param, self.scale ),
                         bias = self.bias_param,
                         stride = self.stride,
                         padding = self.padding
                        )

class ToRGB( nn.Module ):
    def __init__(self, in_channels, out_channels ):
        super(ToRGB, self).__init__()
        self.equalized_conv1 = EqualizedConv2d( in_channels, out_channels, 1, 1, 0 )
    def forward(self, net):
        return self.equalized_conv1( net )


class FromRGB( nn.Module ):
    def __init__(self, in_channels, out_channels ):
        super(FromRGB, self).__init__()
        self.equalized_conv1 = EqualizedConv2d( in_channels, out_channels, 1,1, 0  )
        self.leaky_relu1 = nn.LeakyReLU( 0.2 )
    def forward( self, net ):
        net = self.equalized_conv1( net )
        net = self.leaky_relu1( net )
        return net

class GeneratorInitialBlock(nn.Module):
    def __init__( self, in_channels, out_channels, size ):
        super( GeneratorInitialBlock, self ).__init__()
        # here in_channels is the dimension of latent vector
        self.in_channels = in_channels  
        self.out_channels = out_channels
        self.size = size

        self.ln1 = nn.Linear( in_channels ,  out_channels* size *size  )
        self.leaky_relu1 = nn.LeakyReLU(0.2)
        self.equalized_conv1 = EqualizedConv2d( out_channels, out_channels, 3, 1, 1 )
        self.leaky_relu2 = nn.LeakyReLU(0.2)
        self.pixelnorm = PixelNorm()
    def forward( self, net ):
        net = self.ln1(net)
        net = self.leaky_relu1(net).view( -1, self.out_channels, self.size, self.size )
        net = self.equalized_conv1( net )
        net = self.leaky_relu2(net)
        net = self.pixelnorm(net)
        return net

class GeneratorBlock( nn.Module ):
    def __init__(self, in_channels, out_channels, size   ): 
        super( GeneratorBlock, self ).__init__()
        self.size = size
        self.equalized_conv1 = EqualizedConv2d( in_channels, out_channels, 3, 1, 1 )
        self.leaky_relu1 = nn.LeakyReLU(0.2)
        self.pixelnorm1 = PixelNorm()
        self.equalized_conv2 = EqualizedConv2d( out_channels, out_channels, 3, 1, 1 )
        self.leaky_relu2 = nn.LeakyReLU(0.2)
        self.pixelnorm2 = PixelNorm()
    def forward( self, net ):
        net = F.interpolate(net, self.size)
        net = self.equalized_conv1(net)
        net = self.leaky_relu1(net)
        net = self.pixelnorm1(net)
        net = self.equalized_conv2(net)
        net = self.leaky_relu2(net)
        net = self.pixelnorm2(net)
        return net


class DiscriminatorInitialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size ):
        super( DiscriminatorInitialBlock, self ).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.size = size

        self.minibatch_std_dev = MinibatchStddev()
        ## here is minibatch_std_dev, so the channel num will be plus one
        self.equalized_conv1 = EqualizedConv2d( in_channels+1, in_channels, 3,1,1 )
        self.leaky_relu1 = nn.LeakyReLU(0.2)
        self.ln1 = nn.Linear( in_channels* size * size , out_channels  )
        self.leaky_relu2 = nn.LeakyReLU(0.2)
        self.ln2 = nn.Linear( out_channels , 1 )
    def forward(self, net):
        net = self.minibatch_std_dev( net )
        net = self.equalized_conv1(net)
        net = self.leaky_relu1(net)
        net = net.view( -1, self.in_channels* self.size * self.size )
        net = self.ln1(net)
        net = self.leaky_relu2(net)
        net = self.ln2(net)
        return net


class DiscriminatorBlock(nn.Module):
    def __init__(self,  in_channels, out_channels ):
        super( DiscriminatorBlock, self ).__init__()
        self.equalized_conv1 = EqualizedConv2d( in_channels, in_channels, 3,1,1  )
        self.leaky_relu1 = nn.LeakyReLU(0.2)
        self.equalized_conv2 = EqualizedConv2d( in_channels, out_channels, 3,1,1  )
        self.leaky_relu2 = nn.LeakyReLU(0.2)
        self.pooling = nn.AvgPool2d( 2,2 )
    def forward( self, net ):
        net = self.equalized_conv1(net)
        net = self.leaky_relu1(net)
        net = self.equalized_conv2(net)
        net = self.leaky_relu2(net)
        net = self.pooling(net)
        return net

class EncoderInitialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, z_dim , size ):
        super( EncoderInitialBlock, self ).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.size = size

        self.equalized_conv1 = EqualizedConv2d( in_channels, in_channels, 3,1,1 )
        self.leaky_relu1 = nn.LeakyReLU(0.2)
        self.ln1 = nn.Linear( in_channels* size * size , out_channels  )
        self.leaky_relu2 = nn.LeakyReLU(0.2)
        self.ln2 = nn.Linear( out_channels , z_dim )
    def forward(self, net):
        net = self.equalized_conv1(net)
        net = self.leaky_relu1(net)
        net = net.view( -1, self.in_channels* self.size * self.size )
        net = self.ln1(net)
        net = self.leaky_relu2(net)
        net = self.ln2(net)
        return net


class EncoderBlock(nn.Module):
    def __init__(self,  in_channels, out_channels ):
        super( EncoderBlock, self ).__init__()
        self.equalized_conv1 = EqualizedConv2d( in_channels, in_channels, 3,1,1  )
        self.leaky_relu1 = nn.LeakyReLU(0.2)
        self.equalized_conv2 = EqualizedConv2d( in_channels, out_channels, 3,1,1  )
        self.leaky_relu2 = nn.LeakyReLU(0.2)
        self.pooling = nn.AvgPool2d( 2,2 )
    def forward( self, net ):
        net = self.equalized_conv1(net)
        net = self.leaky_relu1(net)
        net = self.equalized_conv2(net)
        net = self.leaky_relu2(net)
        net = self.pooling(net)
        return net