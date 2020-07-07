# Generation and Analysis of Art using Machine Learning
This repository is the code for our Final Year Thesis.
[Video Demo](https://youtu.be/47BCm7O7S8c)

## Project Description
Novel art generation, with focus on religious art, using different versions of Generative Adversarial Networks (GANs) implemented with various GAN-optimization techniques. Various novel techniques such as "glitching" and "watermarking" were used to make it easier for the low dimensional GAN to pick the trends out of the low resolution images.

## Pipeline


![pipeline](https://github.com/ShahzaibWaseem/ArtGAN/blob/master/Images/pipeline.png)

## Techniques Used

We employed all techniques suggested by [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498) Paper and [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028) Paper. Apart from these techniques we employed some novel techniques which make it easier for the low dimensional GAN to capture trends from low resolution images, which are shown below.

### Watermarker Script
Gives an easy trend for the GAN to capture.


![watermark](https://github.com/ShahzaibWaseem/ArtGAN/blob/master/Images/Watermarker.png)

### Glitcher Script
Makes the whole image easier for the GAN to understand.


![glitcher](https://github.com/ShahzaibWaseem/ArtGAN/blob/master/Images/Glitcher.png)

## Loss
The following image shows how the loss function behaves when we use WGAN-gp for 2500 epochs and Batch size 32.


![WGAN-gp-2500epochs,32batch](https://github.com/ShahzaibWaseem/ArtGAN/blob/master/Images/WGAN_Batch32.jpeg)


## Copyright
The Fast SR-GAN module was forked from [Fast-SRGAN](https://github.com/HasnainRaz/Fast-SRGAN) repository.