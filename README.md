# Motion Deblurring U-Net and Ablation Study

![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ml-matthew-lam/deblurring_ablation/blob/main/train.ipynb)

In this project, I built a nonlinear-activation-free (NAF) U-Net architecture for motion deblurring. I then implemented a combined L1 and SSIM loss function and trained the model on the GOPRO_Large dataset, achieving an average 27.2 PSNR and 0.832 SSIM on the test set.

I also ran an ablation study testing the effect of replacing the SimpleGate operations with GELU over shorter training durations. The NAF model outperformed the architecture with GELU activations (+0.75 PSNR, +0.023 SSIM), which is consistent with literature.

See more about this project on <a href="https://matthewlam.me/deblurring/" target="_blank">my website</a>.


## Results

| Architecture Variant | Activation | SSIM | PSNR (dB) |
|----------------------|------------|-----------|------|
| 500-epoch NAF model | SimpleGate | ??? | ??? |
| 200-epoch NAF model | GELU | 0.8318 | 27.18|
| 200-epoch GELU model | SimpleGate | 0.8088 | 26.43|
| Blurry inputs | n/a | 0.7733 | 25.61|


## Repository Structure

The codebase is modularized to separate the architecture definitions from the training and testing loops:
* **`model.py`**: Contains the core U-Net architecture with modular support for both NAF and GELU blocks.
* **`loss.py`**: Custom combined loss function (L1 + SSIM), including a specialized 2D window implementation for SSIM calculations.
* **`dataset.py`**: PyTorch Dataset class for the GOPRO_Large dataset, handling image cropping, flipping augmentations, and train/val/test splits.
* **`train.ipynb`**: The primary training notebook containing the optimization loop, learning rate scheduling, and Weights & Biases (wandb) logging.
* **`test.py`**: Evaluation script to benchmark model variants, calculate peak system RAM usage/inference time, and generate comparative image triplets (Blurry vs. Restored vs. Sharp).
  

## Setup and Installation

**1. Clone the repository and install dependencies:**
```bash
git clone [https://github.com/ml-matthew-lam/deblurring_ablation.git](https://github.com/ml-matthew-lam/deblurring_ablation.git)
cd deblurring_ablation
pip install -r requirements.txt
```

**2. Download the Dataset:**
Download the GOPRO_Large.zip dataset. Extract it so that the GOPRO_Large folder sits at the root of the project directory (or adjust the paths in test.py accordingly).  

## Training in Google Colab

This project is optimized for training on GPU hardware via Google Colab. 
Click the Open in Colab badge at the top of this README and follow the instruction in the notebook.
(Note: The notebook handles the extraction of the dataset and cloning of this repository automatically).

## Running Inference & Evaluation

To benchmark a trained model locally, place your saved weights (e.g., NAF_best.pth) in the root directory and run the testing script:
```bash
python test.py
```
This script will evaluate the model against the test set, output a metrics.json file with inference statistics, and save visual comparisons of images of deblurring examples.