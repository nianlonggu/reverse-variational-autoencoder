from model import *
from utils import *

import argparse


class Flags:
    def __init__(self, ngpu, model_dir, train_data_dir, test_data_dir ):
        self.ngpu = ngpu

        self.model_dir = model_dir+"/"
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
    parser.add_argument("--model_dir" )
    parser.add_argument("--train_data_dir" )
    parser.add_argument("--test_data_dir" )
    args = parser.parse_args()
    ngpu = args.ngpu
    model_dir = args.model_dir
    train_data_dir = args.train_data_dir
    test_data_dir = args.test_data_dir

    FLAGS = Flags( ngpu, model_dir, train_data_dir, test_data_dir)

    generator = Generator( FLAGS.z_dim, FLAGS.generator_out_channels_list, FLAGS.image_size_list, FLAGS.image_color_channels )
    discriminator = Discriminator( FLAGS.discriminator_in_channels_list, FLAGS.discriminator_out_channels_list, FLAGS.image_size_list, FLAGS.image_color_channels )
    encoder = Encoder( FLAGS.encoder_in_channels_list, FLAGS.encoder_out_channels_list, FLAGS.image_size_list, FLAGS.image_color_channels, FLAGS.z_dim )
    status = Status()

    generator_ema = Generator( FLAGS.z_dim, FLAGS.generator_out_channels_list, FLAGS.image_size_list, FLAGS.image_color_channels )
    discriminator_ema = Discriminator( FLAGS.discriminator_in_channels_list, FLAGS.discriminator_out_channels_list, FLAGS.image_size_list, FLAGS.image_color_channels )
    encoder_ema = Encoder( FLAGS.encoder_in_channels_list, FLAGS.encoder_out_channels_list, FLAGS.image_size_list, FLAGS.image_color_channels, FLAGS.z_dim )
    
    ckpt_file_names = glob.glob( FLAGS.model_dir+"inference/*.pt" )
    ckpt_file_names.sort( key= os.path.getmtime )
    if len(ckpt_file_names)>0:
        load_model({"gen":generator, 
                "disc":discriminator, 
                "enc":encoder,  
                "status": status},  ckpt_file_names[-1]  )  # load the most recent model
                
    generator = generator.to(FLAGS.device)
    discriminator = discriminator.to(FLAGS.device)
    encoder = encoder.to( FLAGS.device )

    generator_ema = generator_ema.to(FLAGS.device)
    discriminator_ema = discriminator_ema.to(FLAGS.device)
    encoder_ema = encoder_ema.to(FLAGS.device)
    
    update_moving_average( generator_ema, generator, decay = 0 )
    update_moving_average( discriminator_ema, discriminator, decay = 0 )
    update_moving_average( encoder_ema, encoder, decay = 0 )

    if FLAGS.device.type=="cuda" and  FLAGS.ngpu >1:
        generator = nn.DataParallel( generator, list( range( FLAGS.ngpu ) )  )
        discriminator = nn.DataParallel( discriminator, list( range(FLAGS.ngpu) ) )
        encoder = nn.DataParallel(encoder , list( range(FLAGS.ngpu) ))

    minus_gradient = torch.tensor( -1, dtype= torch.float32, device= FLAGS.device )  # this variable is used to controll the minus gradient
    
    fixed_noise = torch.randn( 8, FLAGS.z_dim, device= FLAGS.device ) # This fixed nose is used to visualize the progress of training (generated images)
    
    ## This part is used to visualize the reconstruction performance of the model
    test_image_loader = CustomImageLoader( FLAGS.test_data_dir, FLAGS.image_size_list[-1], 8 , FLAGS.image_color_channels )
    test_real_x_batch = test_image_loader.get_next()[0].to(FLAGS.device)
    test_real_x_for_stage = []
    for stg in range( len(FLAGS.image_size_list) -1 ):
        test_real_x_for_stage.append( F.interpolate( test_real_x_batch, FLAGS.image_size_list[stg] ) )
    test_real_x_for_stage.append( test_real_x_batch )

    start_training_stage = status.current_stage

    for stage in range( start_training_stage,  len( FLAGS.image_size_list )  ):
        ## update status for record
        status.current_stage = stage

        image_loader = CustomImageLoader( FLAGS.train_data_dir, FLAGS.image_size_list[stage], FLAGS.batch_size_list[stage], FLAGS.image_color_channels )
        
        if FLAGS.device.type=="cuda" and  FLAGS.ngpu >1:
            D_models = list(discriminator.module.disc_blocks[ :stage+1 ])+ list( discriminator.module.fromRGB_blocks[ max(0, stage-1): stage +1 ])
            G_models = list(generator.module.generator_blocks[ :stage+1 ]) + list(generator.module.toRGB_blocks[ max(0,stage-1): stage+1 ])    
            E_models = list(encoder.module.encoder_blocks[ : stage+1] ) + list( encoder.module.fromRGB_blocks[ max(0, stage-1): stage+1  ] )
        else:
            D_models = list(discriminator.disc_blocks[ :stage+1 ])+ list( discriminator.fromRGB_blocks[ max(0, stage-1): stage +1 ])
            G_models = list(generator.generator_blocks[ :stage+1 ]) + list(generator.toRGB_blocks[ max(0,stage-1): stage+1 ])
            E_models = list(encoder.encoder_blocks[ : stage+1] ) + list( encoder.fromRGB_blocks[ max(0, stage-1): stage+1  ] )
        
        optimizerD = optim.Adam( [par for model in D_models for par in model.parameters()]  , lr=1e-3, betas=(0, 0.99))
        optimizerG = optim.Adam( [par for model in G_models for par in model.parameters()]  , lr=1e-3, betas=(0, 0.99)) 
        optimizerE = optim.Adam( [par for model in E_models for par in model.parameters()]  , lr=1e-3, betas=(0, 0.99) )

        if stage == start_training_stage:
            start_training_step = status.current_step
        else:
            start_training_step = 0

        prev_time = time.time()
        prev_step = start_training_step
        for step in range(start_training_step, FLAGS.train_steps_list[stage] ):
            status.current_step = step
           
            for model in D_models:
                for p in model.parameters():
                    p.requires_grad = True
            ## compute alpha
            if stage == 0:
                alpha = 1
            else:
                alpha = max(min(  step / ( FLAGS.train_steps_list[stage]/2) , 1.0 ), 0.0 )  
            status.current_alpha = alpha

            for iter_d in range( FLAGS.n_critic ):
                for model in D_models:
                    model.zero_grad()   

                real_x = image_loader.get_next()[0].to(FLAGS.device)
                
                if stage >0 and alpha < 1:
                    ## here real_x should be resampled to simulate the same procedure as the generator's output
                    real_x_low = F.interpolate( real_x, FLAGS.image_size_list[stage-1] )
                    real_x_low_up = F.interpolate( real_x_low, FLAGS.image_size_list[stage] )
                    real_x = alpha * real_x + (1-alpha) * real_x_low_up

                dis_x_raw = discriminator( real_x, stage, alpha  )
                dis_x = dis_x_raw.view(-1).mean()
                # retain_graph= True since in the next we need to backward agian on the same graph
                dis_x.backward(minus_gradient, retain_graph=True)   

                dis_drift_loss =  FLAGS.e_drift * dis_x_raw.pow(2).view(-1).mean()
                dis_drift_loss.backward()   

                z = torch.randn( FLAGS.batch_size_list[stage], FLAGS.z_dim, device= FLAGS.device )
                gen_x =  generator( z,  stage, alpha).detach()
                dis_gen_x = discriminator( gen_x, stage, alpha )
                dis_gen_x = dis_gen_x.view(-1).mean()
                dis_gen_x.backward()    

                gradient_penalty = compute_gradient_penalty( real_x.data, gen_x.data, discriminator , stage, alpha, FLAGS.LAMBDA, FLAGS.device )
                gradient_penalty.backward() 

                D_cost = dis_gen_x - dis_x + gradient_penalty + dis_drift_loss
                Wasserstein_D = dis_x - dis_gen_x
                optimizerD.step()
            
            for model in D_models + E_models :
                for p in model.parameters():
                    p.requires_grad = False
            for model in G_models:
                model.zero_grad()   

            z = torch.randn( FLAGS.batch_size_list[stage], FLAGS.z_dim, device= FLAGS.device )
            gen_x = generator( z,  stage, alpha)
            dis_gen_x = discriminator( gen_x, stage, alpha )
            dis_gen_x = dis_gen_x.view(-1).mean()
            dis_gen_x.backward(minus_gradient, retain_graph = True)
            G_cost = -dis_gen_x

            recon_z = encoder( gen_x, stage, alpha )
            loss_recon_z =  (z - recon_z).pow(2).sum(dim=1).mean() + torch.abs( recon_z.std() -1)
            loss_recon_z.backward( torch.tensor( 1, dtype = torch.float32, device=FLAGS.device)  )
            optimizerG.step()  

            for model in E_models :
                for p in model.parameters():
                    p.requires_grad = True
                model.zero_grad()   
            z = torch.randn( FLAGS.batch_size_list[stage], FLAGS.z_dim, device= FLAGS.device )
            gen_x = generator( z, stage, alpha ).detach()
            recon_z = encoder( gen_x, stage, alpha )
            loss_recon_z =  (z - recon_z).pow(2).sum(dim=1).mean() +  torch.abs( recon_z.std() -1)
            loss_recon_z.backward()
            optimizerE.step()

            update_moving_average( generator_ema, generator, decay = 0.999 )
            update_moving_average( discriminator_ema, discriminator, decay = 0.999 )
            update_moving_average( encoder_ema, encoder, decay = 0.999 )

            if step % 200 == 199:
                current_time = time.time()
                
                # print( step, "W_dis:", Wasserstein_D.cpu() )
                # print( "%d images per second" % (  (step-prev_step) * FLAGS.batch_size_list[stage]/ ( current_time - prev_time  )  ) )
                os.system( "nvidia-smi >> monitor_gpu.txt" )
                print_info = "Step: %d, W_dis: %.4f, %d images per second\n" %( step, Wasserstein_D.cpu(),  (step-prev_step) * FLAGS.batch_size_list[stage]/ ( current_time - prev_time  )  )
                
                with open( "training_progress.txt", "a" ) as f:
                    f.write( print_info )

                prev_step = step
                prev_time = current_time

                save_model( {"gen":generator_ema, 
                            "disc":discriminator_ema, 
                            "enc":encoder_ema,  
                            "status": status}, 
                                      FLAGS.model_dir+"inference/model_stage_%d_step_%d.pt"%(stage, step), 5 )
        
                # save_model( {"gen":generator, 
                #             "disc":discriminator, 
                #             "enc":encoder,  
                #             "status": status}, 
                #                       FLAGS.model_dir+"inference/model_stage_%d_step_%d.pt"%(stage, step), 5 )
        

        os.system( "nvidia-smi >> monitor_gpu.txt" )
        # save_model( {"gen":generator_ema, 
        #             "disc":discriminator_ema, 
        #             "enc":encoder_ema,  
        #             "status": status  }, 
        #                         FLAGS.model_dir+"stagecheckpoint/model_stage_%d_step_%d.pt"%(stage, step), 1 )

        save_model( {"gen":generator_ema, 
                    "disc":discriminator_ema, 
                    "enc":encoder_ema,  
                    "status": status  }, 
                                FLAGS.model_dir+"stagecheckpoint/model_stage_%d_step_%d.pt"%(stage, step), 0 )


        fake = generator( fixed_noise, stage).detach()
        fake = fake.clamp( -1,1 )/2+0.5
        # plt.figure(figsize=(8,8))
        # plt.axis("off")
        # plt.title("generation visualization")
        # plt.gray()
        img = np.transpose(  make_grid(fake , normalize=True).cpu() ,[1,2,0]   ).numpy()
        # plt.imshow(img)
        # plt.show()
        imageio.imsave( "stage_%d_generated.png"%( stage ), img )     

        ## detach is required since we want requires_grad = False
        test_recon = generator( encoder( test_real_x_for_stage[stage] , stage )  , stage).detach()
        # plt.figure(figsize=(8,8))
        # plt.axis("off")
        # plt.title("generation visualization")
        # plt.gray()
        img = np.transpose(  make_grid( torch.cat( [ test_real_x_for_stage[stage].clamp( -1,1 )/2+0.5,  test_recon.clamp( -1,1 )/2+0.5 ], dim = 3 )  , normalize=True).cpu() ,[1,2,0]   ).numpy()
        # plt.imshow(img)
        # plt.show()
        imageio.imsave( "stage_%d_recon.png"%( stage ), img ) 


