# Reverse Variational Autoencoder for Visual Attributes Manipulation and Anomaly Detection
## Data Preparation
* Extract the celeba HQ images using the code from https://github.com/willylulu/celeba-hq-modified
* Split the 30K celeba 1024x1024 images into training set and test set. The training images are placed in the folder "dataset/celeba/train/image/" while the testing images are placed in the folder "dataset/celeba/test/image/"

## Training
```
python train.py  --ngpu 8 --model_dir model/ --train_data_dir dataset/celeba/train/ --test_data_dir dataset/celeba/test/
```
In the file train.py you can configure the hyperparameters, including the model size and final image resolutions. The default image resolsution is 1024x1024, which means the model will gradually grow from 4x4, 8x8, ... finally to 1024x1024. When training the 1024x1024 model, 8 x Tesla V100 32GB GPUs were used. However, if the final resolustion is 512x512, the model should be able to be trained on 8 x GTX 1080Ti GPUs (not verified).
## Testing
```
python inference.py  --ngpu 8 --ckpt_file_name model/stagecheckpoint/model_stage_#_step_#.pt --train_data_dir dataset/celeba/train/ --test_data_dir dataset/celeba/test/
```
You can specify the stage number and step number to determine which resolustion stage in which step you are going to test.

## Inference Results
### Random Generated Images
8x8  : ![8x8](https://github.com/nianlonggu/reverse_variational_autoencoder/blob/master/results/8x8_gen.png)

16x16: ![16x16](https://github.com/nianlonggu/reverse_variational_autoencoder/blob/master/results/16x16_gen.png)

32x32: ![32x32](https://github.com/nianlonggu/reverse_variational_autoencoder/blob/master/results/32x32_gen.png)

64x64: ![64x64](https://github.com/nianlonggu/reverse_variational_autoencoder/blob/master/results/64x64_gen.png)

128x128: ![128x128](https://github.com/nianlonggu/reverse_variational_autoencoder/blob/master/results/128x128_gen.png)

256x256: ![256x256](https://github.com/nianlonggu/reverse_variational_autoencoder/blob/master/results/256x256_gen.png)

512x512: ![512x512](https://github.com/nianlonggu/reverse_variational_autoencoder/blob/master/results/512x512_gen.png)

### Image Reconstructions
8x8  : ![8x8](https://github.com/nianlonggu/reverse_variational_autoencoder/blob/master/results/8x8_recon.png)

16x16: ![16x16](https://github.com/nianlonggu/reverse_variational_autoencoder/blob/master/results/16x16_recon.png)

32x32: ![32x32](https://github.com/nianlonggu/reverse_variational_autoencoder/blob/master/results/32x32_recon.png)

64x64: ![64x64](https://github.com/nianlonggu/reverse_variational_autoencoder/blob/master/results/64x64_recon.png)

128x128: ![128x128](https://github.com/nianlonggu/reverse_variational_autoencoder/blob/master/results/128x128_recon.png)

256x256: ![256x256](https://github.com/nianlonggu/reverse_variational_autoencoder/blob/master/results/256x256_recon.png)

512x512: ![512x512](https://github.com/nianlonggu/reverse_variational_autoencoder/blob/master/results/512x512_recon.png)

## Paper
Lydia, Gauerhof, and Nianlong Gu. "Reverse Variational Autoencoder for Visual Attribute Manipulation and Anomaly Detection." The IEEE Winter Conference on Applications of Computer Vision. 2020.

## Contact
If you have any question, please contact nianlonggu@gmail.com or Lydia.Gauerhof@de.bosch.com
