# Assignment 5: Autoregressive image completion
In this assignment you will learn to implement a basic autoregressive model for completing an image. The model is based on the PixelCNN described in the paper [Pixel Recurrent Neural Networks, Oord et al.](https://arxiv.org/pdf/1601.06759).

---

## Overview

The basic idea of the method is to synthesize pixels one-by-one starting from the top left corner in an image region and proceeding in a row major order.

The convnet you will implement is small, the dataset is small, and can be trained on the CPU (i.e., training takes ~10-15 minutes, and testing takes a few seconds per image in the instuctor's 5-year old laptop with a i7-5950HQ).  If you want to use a GPU (this is not required by this assignment), you need to have access to a GPU graphics card with at least 2GB memory.

Since the dataset is very small, different executions of training may lead to different local minima with somewhat different results. You may repeat the training a few times to select the best model (i.e., the one that gives the best visual results at test time). A few examples of desired results are shown below. The odd columns show test images corrupted with salt-and-pepper noise in a small window near the center of the image. The even columns show expected restorations (the average test reconstruction error in terms of L1 error should be below 0.1). Ideally, one would want to use much larger training datasets for training image synthesis/completion methods, and also use more modern and complex autoregressive variants (e.g., see VQGAN, ASSET, ImageBART etc) based on transformers or diffusion models, which are all beyond the scope of this assignment.

## Dataset

1. Training dataset of 2K 100x100 chair images
2. Small validation split of 96 images
3. Test split of 40 images

---

# Tasks

## Task A (30%)

---

Change the starter code in "model.py" to define the following convnet with three blocks each consisting of:
1. A masked convolution layer with 16 filters applying 3x3 filters on the input image with a stride of 1, a dilation of 3 to enlarge the receptive field of each filter, and padding to 'reflect'. The masked convolution zero-es out weights for the bottom right part of the pixel, as described in class and the paper.
2. A batch normalization layer
3. A leaky ReLU layer with negative slope of 0.001

The three blocks share the same structure as above (still with their own filter weights at each convolution layer). The masked convolution in the first layer should use a mask which zero-es out the weight at the center of the filters. The rest of the masked convolution layers should not mask the center. Use the MaskedCNN class appropriately while defining your model in PixelCNN.

After the above 3 blocks, a 1x1 convolution layer processes the 16 channels from the last block, and outputs a single channel (ie the grayscale intensity for the output image). Set the padding size in the convolution layers such that the resulting channel has the same height and width with the input image. The channel intensities should then be squeezed between 0 and 1 through a sigmoid function. Think about whether you should use the bias in each of the convolutional layers.

For all layer parameters, use the default PyTorch initialization scheme (no need to write any code for initializing the model parameters).

---

## Task B (30%)

---

Change the starter code in "trainARimage.py" to train your network such that it predicts an output image that matches the input image. Training should be done using an output image that matches the input image. Training should be done using the L1 loss function (already implemented as nn.L1loss in PyTorch).

Complete the code such that:
1. Each batch is loaded correctly along with the desired targets.
2. Backpropagation is executed to update the model parameters.
3. The L1 loss is computed for each batch.
4. The training and validation loss are computed for each epoch.

---

## Task C (30%)

---

Change the starter code in "testARimage.py" to correct each pixel in the distorted region in an autoregressive manner. This involves implementing a loop that predicts each pixel starting at the top left of the distorted region and proceeding in a row major order. You should not change pixels outside the distorted region. The starter code already computes the L1 error for each test image.

---

## Task D (10%)

---

Write a README.txt reporting the average reconstruction error you achieved for the best model you trained.
