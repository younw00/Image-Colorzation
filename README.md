# Image Colorization
## ✨Motivation
This is a collaborative team project centered on employing advanced computer vision techniques to effectively colorize grayscale images.
## ✨Skills
- Python
- PyTorch
## ✨Work with
🧑‍💻 DongSub Kim 👩‍💻 Younwoo Yim 🧑‍💻 Zachary Eichenberger
## ✨Description
### Dataset
Large-scale images were downloaded from ImageNet which contains about 20 thousand categories of images for our classification implementation. To make the project computationally feasible, we used only a tiny portion of ImageNet which contains 200 categories of 64x64 pixel images. 
10k images are used to train the model and each 3k image is used in the evaluation sets and in the test sets.
### Model
We treated the problem as a regression problem, having our model develop the a and b channels of a LAB colorspace, given just the luminance dimension. Then at evaluation time, we re-combined the predicted channels with
the actual luminance values to produce a final output. Our model took a (1 x 64 x 64) image, and output a (2 x 64 x64) image representing colorspace estimates for each pixelin that image. We developed two models, a simple model, and a U net.

<img src="https://github.com/younw00/Image-Colorzation/assets/107108235/ca6ed309-b4ca-4b57-a374-62057928247c">
