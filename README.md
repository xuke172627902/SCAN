models and codes for SCAN

The code is constructed based on BasicSR, Multi-scale strip-shaped convolution attention network for lightweight image super-resolution (MSAN). Before any testing or reproducing, make sure the installation and the datasets preparation are done correctly.

To keep the workspace clean and simple, only SCAN_for_distillation.py, hat_block_disitllation.py and train_semi_online_distillation.py are needed here and then you are good to go.(hat.py is the original code of HAT(CVPR2024).
To make hat_block_distillation explainable you can refer to the official implementation of HAT)

environment：

Python >= 3.8.0

Pyotch >= 1.8.1

torchvision >=0.16.1

basicsr = 1.4.2

dataset:

train_Data:

DIV2K(800 images for training and 100 images for validation)

Flicker2K(2650 images)

test_data:

Set5, Set14, BSDS100, Urban100, Manga109

All datasets could be found in https://paperswithcode.com/datasets.

More preparation for training datasets:

See https://github.com/XPixelGroup/BasicSR/tree/master/basicsr/data for more details. The document .pth of HAT is avaiable at https://github.com/XPixelGroup/HAT

Training and testing:

hyperparameters like training iterations, learning rate, optim options, disitllation interval could be adjusted in train_semi_online_distillation.py. For more settings please refer to our paper:
Strip-shaped convolutional attention network for efficient image super-resolution via feature-based semi-online distillation

