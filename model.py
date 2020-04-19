from networks import *


class Generator(nn.Module):
    def __init__( self, z_dim, out_channels_list, size_list, image_color_channels ):
        super( Generator, self ).__init__()        

        self.size_list = size_list
        num_blocks = len(out_channels_list)
        ## create the generator blocks and put them into MoudleList, so that these modules can be added to the graph and can be trained
        generator_blocks = []
        generator_blocks.append( GeneratorInitialBlock( z_dim, out_channels_list[0], size_list[0]   ) )
        for stage in range(1, num_blocks ):
            generator_blocks.append( GeneratorBlock( out_channels_list[stage-1], out_channels_list[stage], size_list[stage] ) )
        self.generator_blocks = nn.ModuleList( generator_blocks )
        
        ## create toRGB blocks and put them into MoudleList
        toRGB_blocks = []
        for stage in range( num_blocks ):
            toRGB_blocks.append( ToRGB( out_channels_list[stage], image_color_channels ) )
        self.toRGB_blocks = nn.ModuleList( toRGB_blocks )

    def forward( self, z, stage, alpha=1):
        if stage == 0:
            gen_x = self.toRGB_blocks[stage]( self.generator_blocks[stage](z) )
        else:
            net = z
            for stg in range(stage):
                net = self.generator_blocks[stg](net)
            gen_x_1 = F.interpolate(self.toRGB_blocks[stage-1](net), self.size_list[stage] )
            net = self.generator_blocks[stage](net)
            gen_x_2 = self.toRGB_blocks[stage](net)
            gen_x = alpha * gen_x_2 + (1-alpha) * gen_x_1
        return gen_x 


class Discriminator(nn.Module):
    def __init__(self, in_channels_list, out_channels_list, size_list, image_color_channels ):
        super( Discriminator, self ).__init__()
        num_blocks = len(out_channels_list)
        disc_blocks = []
        disc_blocks.append( DiscriminatorInitialBlock( in_channels_list[0], out_channels_list[0], size_list[0] ) )
        for stage in range(1, num_blocks ):
            disc_blocks.append( DiscriminatorBlock( in_channels_list[ stage], out_channels_list[stage]  ) )
        self.disc_blocks = nn.ModuleList( disc_blocks )

        fromRGB_blocks = []
        for stage in range( num_blocks ):
            fromRGB_blocks.append( FromRGB( image_color_channels, in_channels_list[ stage] ) )
        self.fromRGB_blocks = nn.ModuleList( fromRGB_blocks )
    
    def forward( self, x, stage, alpha = 1):
        if stage == 0:
            dis_x = self.disc_blocks[stage]( self.fromRGB_blocks[stage](x)  )
        else:
            net_1 = self.disc_blocks[stage]( self.fromRGB_blocks[stage](x)  )
            net_2 = self.fromRGB_blocks[stage-1]( F.avg_pool2d( x,2,2 ) )
            net = alpha * net_1 + (1-alpha) * net_2
            for stg in range( stage-1, -1, -1 ):
                net = self.disc_blocks[stg](net)
            dis_x = net
        return dis_x

class Encoder(nn.Module):
    def __init__(self, in_channels_list, out_channels_list, size_list, image_color_channels, z_dim ):
        super( Encoder, self ).__init__()
        num_blocks = len(out_channels_list)
        encoder_blocks = []
        encoder_blocks.append( EncoderInitialBlock( in_channels_list[0], out_channels_list[0], z_dim , size_list[0] ) )
        for stage in range(1, num_blocks ):
            encoder_blocks.append( EncoderBlock( in_channels_list[ stage], out_channels_list[stage]  ) )
        self.encoder_blocks = nn.ModuleList( encoder_blocks )

        fromRGB_blocks = []
        for stage in range( num_blocks ):
            fromRGB_blocks.append( FromRGB( image_color_channels, in_channels_list[ stage] ) )
        self.fromRGB_blocks = nn.ModuleList( fromRGB_blocks )
    
    def forward( self, x, stage, alpha = 1):
        if stage == 0:
            enc_x = self.encoder_blocks[stage]( self.fromRGB_blocks[stage](x)  )
        else:
            net_1 = self.encoder_blocks[stage]( self.fromRGB_blocks[stage](x)  )
            net_2 = self.fromRGB_blocks[stage-1]( F.avg_pool2d( x,2,2 ) )
            net = alpha * net_1 + (1-alpha) * net_2
            for stg in range( stage-1, -1, -1 ):
                net = self.encoder_blocks[stg](net)
            enc_x = net
        return enc_x










