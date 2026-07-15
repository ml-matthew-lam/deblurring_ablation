# Image Deblurring U-Net and Ablation Study   ![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
In this project, I built a nonlinear-activation-free (NAF) U-Net architecture for motion deblurring. I then implemented a combined L1 and SSIM loss function and trained the model on the GOPRO_Large dataset, achieving an average 27.2 PSNR and 0.832 SSIM on the test set. I also tested the effect of replacing the SimpleGate operations with GELU over shorter training durations. The NAF model outperformed the architecture with GELU activations (+0.75 PSNR, +0.023 SSIM), which is consistent with literature.
See more about this project on [my website](https://matthewlam.me/deblurring/).

## Results
| Architecture Variant | Activation | SSIM | PSNR (dB) |
|----------------------|------------|-----------|------|
| 500-epoch NAF model | SimpleGate   | ???   |  ??? |
| 200-epoch NAF model  | GELU        | 0.8318   | 27.18|
| 200-epoch GELU model | SimpleGate | 0.8088    | 26.43|
| Blurry inputs       | n/a         | 0.7733    | 25.61|


## Files in this repository
- [model.py](model.py): contains the U-net architecture with the ability to use either NAF and GELU blocks
- [loss.py](loss.py): combined (L1 with SSIM) loss function, including an implementation of SSIM calculations
- [dataset.py](dataset.py): code to prepare training, validation and test sets, and perform data augmentations (cropping and flipping)
- [train.ipynb](train.ipynb): notebook containing code to train the models
- [test.py](test.py): script to test the models and record testing metrics

