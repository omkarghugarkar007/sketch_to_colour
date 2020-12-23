# sketch_to_colour

This repository contains implementation of the CVPR - 2017 Paper "Image-to-Image Translation with Conditional Adversarial Networks". 

### Generator 

The architecture is of U-Net kind, which takes input as a sketch images of size (256 X 256 X 3) and outputs a coloured image of size (256 X 256 X 3). Encoder layer consists of 8 layers which convert images into latent space of size (1 X 1 X 512). Decoder contains 7 layer which upsamples the image.
L1 loss is used to the generator along with the cGan loss

### Discriminator 

It takes sketch as well as coloured images a input and stack one on another. It return the probability that given the sketch, does the coloured image belongs to it or not.

### Training

Gan Model contains the generator and the discriminator architecture. Train contains loss function and the code to train the model. 
### Test
To test the model, run the command,

```
python3 test.py

```