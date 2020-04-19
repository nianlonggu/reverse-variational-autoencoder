from model import *
from utils import *

import argparse


class Flags:
    def __init__(self, ngpu,  train_data_dir, test_data_dir ):
        self.ngpu = ngpu

        self.train_data_dir = train_data_dir + "/" 
        self.test_data_dir = test_data_dir +"/" 
        self.device = torch.device( "cuda:0" if torch.cuda.is_available() and self.ngpu>0 else "cpu"  )
        
        self.z_dim = 512
        self.image_color_channels  = 3
        self.n_critic = 1
        self.LAMBDA = 10
        self.e_drift = 1e-3

        self.num_images_per_stage = 800000
        self.batch_size_list = [ 16, 16, 16, 16, 16, 8, 8, 4, 3 ]
        if self.ngpu > 1:
            self.batch_size_list = [ batch_size * self.ngpu for batch_size in self.batch_size_list ]
        self.generator_out_channels_list = [ 512, 512, 512, 512, 256, 128, 64, 32, 16 ]
        self.discriminator_in_channels_list = [ 512, 512, 512, 512, 256, 128, 64, 32, 16 ]
        self.discriminator_out_channels_list = [ 512, 512, 512, 512,  512, 256, 128, 64, 32 ]
        self.encoder_in_channels_list = self.discriminator_in_channels_list
        self.encoder_out_channels_list = self.discriminator_out_channels_list
        self.image_size_list = [4,8,16,32, 64, 128, 256, 512, 1024]
        self.train_steps_list = [ int( self.num_images_per_stage/(self.batch_size_list[0] * self.n_critic ) ) ]+ \
                            [int( self.num_images_per_stage*2/(batch_size * self.n_critic) ) for batch_size in self.batch_size_list[1:] ]

class Status:
    def __init__(self):
        self.current_stage = 0
        self.current_alpha = 1
        self.current_step = 0
    def state_dict(self):
        return self.__dict__
    def load_state_dict( self, state_dict ):
        for key in state_dict.keys():
            setattr( self, key, state_dict[key] )


if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--ngpu", type = int, default= 0)
    parser.add_argument("--ckpt_file_name" )
    parser.add_argument("--train_data_dir", default = "" )
    parser.add_argument("--test_data_dir" )
    args = parser.parse_args()
    ngpu = args.ngpu
    ckpt_file_name = args.ckpt_file_name
    train_data_dir = args.train_data_dir
    test_data_dir = args.test_data_dir

    torch.manual_seed(0)

    FLAGS = Flags( ngpu, train_data_dir, test_data_dir)

    generator = Generator( FLAGS.z_dim, FLAGS.generator_out_channels_list, FLAGS.image_size_list, FLAGS.image_color_channels )
    encoder = Encoder( FLAGS.encoder_in_channels_list, FLAGS.encoder_out_channels_list, FLAGS.image_size_list, FLAGS.image_color_channels, FLAGS.z_dim )
    status = Status()

    
    # ckpt_file_names = glob.glob( FLAGS.model_dir+"inference/*.pt" )
    # ckpt_file_names.sort( key= os.path.getmtime )
    # if len(ckpt_file_names)>0:
    load_model({"gen":generator, 
                # "disc":discriminator, 
                "enc":encoder,  
                "status": status},  ckpt_file_name  )  # load the most recent model
                
    generator = generator.to(FLAGS.device)
    encoder = encoder.to( FLAGS.device )

    if FLAGS.device.type=="cuda" and  FLAGS.ngpu >1:
        generator = nn.DataParallel( generator, list( range( FLAGS.ngpu ) )  )
        encoder = nn.DataParallel(encoder , list( range(FLAGS.ngpu) ))

    
    test_image_loader = CustomImageLoader( FLAGS.test_data_dir, FLAGS.image_size_list[-1], 1 , FLAGS.image_color_channels )

    stage = status.current_stage
    alpha = status.current_alpha

    generated_images_list = []
    input_images_list = []
    reconstructed_images_list = []

    for _ in range(64):
        z = torch.randn( 1, FLAGS.z_dim, device= FLAGS.device ) 
        real_x = F.interpolate( test_image_loader.get_next()[0].to(FLAGS.device) , FLAGS.image_size_list[stage]  )
        if stage >0 and alpha < 1:
            real_x_low = F.interpolate( real_x, FLAGS.image_size_list[stage-1] )
            real_x_low_up = F.interpolate( real_x_low, FLAGS.image_size_list[stage] )
            real_x = alpha * real_x + (1-alpha) * real_x_low_up

        gen_x = generator( z, stage, alpha ).detach()
        recon_x = generator( encoder( real_x, stage, alpha ).detach() , stage, alpha ).detach()

        generated_images_list.append( gen_x )
        input_images_list.append( real_x )
        reconstructed_images_list.append( recon_x )

    generated_images = torch.cat( generated_images_list, dim = 0 ).clamp( -1,1 )/2+0.5
    input_images = torch.cat( input_images_list, dim = 0 ).clamp( -1,1 )/2+0.5
    reconstructed_images = torch.cat( reconstructed_images_list, dim =0 ).clamp( -1,1 )/2+0.5

    gen_img = np.transpose(  make_grid(generated_images , normalize=True).cpu() ,[1,2,0]   ).numpy()
    imageio.imsave( "inference_stage_%d_generated.png"%( stage ), gen_img )     

    recon_img = np.transpose(  make_grid( torch.cat( [ input_images,  reconstructed_images ], dim = 3 )  , normalize=True).cpu() ,[1,2,0]   ).numpy()
    imageio.imsave( "inference_stage_%d_recon.png"%( stage ), recon_img ) 


