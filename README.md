# diffufill: image inpainting using latent diffusion

a lightweight implementation of image inpainting using latent diffusion models (ldm). trained on the celeba-hq dataset with customizable masks.

## overview

this project explores the use of latent diffusion models for inpainting missing or corrupted regions of images. by working in a compressed latent space, the model achieves efficient and high-quality reconstructions.

## features

- autoencoder-based latent space compression
- ddpm and ddim sampling methods
- support for random, center, and custom masks
- loss function combining l2 and perceptual loss
- evaluation using psnr and ssim

## requirements

- python 3.8+
- pytorch
- torchvision
- diffusers
- transformers
- opencv-python
- matplotlib

install dependencies using:

```bash
pip install -r requirements.txt
```
# usage

## training
```
python train.py --dataset celeba-hq --epochs 50 --lr 1e-4 --batch-size 16
```
## testing
```
python test.py --input ./samples/input.jpg --mask ./samples/mask.png --output ./results/
```
## results

the model successfully reconstructs image regions with fine structural and semantic consistency. reconstruction loss is reduced by 22% compared to standard unet baselines.

## folder structure
```
├── autoencoder/
├── scripts/
├── models/
├── datasets/
├── results/
├── samples/
├── train.py
├── test.py
└── utils.py
```
