# Reverse Variational Autoencoder for Visual Attributes Manipulation and Anomaly Detection
## Data Preparation
* Extract the celeba HQ images using the code from https://github.com/willylulu/celeba-hq-modified
* Split the 30K celeba 1024x1024 images into training set and test set. The training images are placed in the folder "dataset/celeba/train/image/" while the testing images are placed in the folder "dataset/celeba/test/image/"

## Training
```
python train.py  --ngpu 8 --model_dir model/ --train_data_dir dataset/celeba/train/ --test_data_dir dataset/celeba/test/
```
In the file train.py you can configure the hyperparameters, including the model size and final image resolutions. The default image resolsution is 1024x1024, which means the model will gradually grow from 4x4, 8x8, ... finally to 1024x1024. When training the 1024x1024 model, 8 x Tesla V100 32GB GPUs were used.
## Testing
```
python inference.py  --ngpu 8 --ckpt_file_name model/stagecheckpoint/model_stage_#_step_#.pt --train_data_dir dataset/celeba/train/ --test_data_dir dataset/celeba/test/
```
You can specify the stage number and step number to determine which resolustion stage in which step you are going to test.

## Inference Results
### 
