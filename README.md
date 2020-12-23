# sketch_to_colour

This repository contains implementation of the CVPR - 2017 Paper "Image-to-Image Translation with Conditional Adversarial Networks". 

### Generator 

The architecture is of U-Net kind, which takes input as a sketch images of size (256 X 256 X 3) and outputs a coloured image of size (256 X 256 X 3). Encoder layer consists of 8 layers which convert images into latent space of size (1 X 1 X 512). Decoder contains 7 layer which upsamples the image.
L1 loss is used to the generator along with the cGan loss

### Discriminator 

It takes sketch as well as coloured images a input and stack one on another. It return the probability that given the sketch, does the coloured image belongs to it or not.

### Training

Gan Model contains the generator and the discriminator architecture. Train contains loss function and the code to train the model. The model was trained on the kaggle dataset Anime-sketch-colorization-pair which contains 14k images. Training was done on Google Colab for 2 session of 8 hours and 100 epochs. 'generator.h5' file contains the generator weights.
### Test
To test the model, run the command,

```
python3 test.py
```

### Results

Some of the results

![image](https://user-images.githubusercontent.com/62425457/103028735-5b023800-457e-11eb-82cd-c585f936b8a8.png)

![image](https://user-images.githubusercontent.com/62425457/103028904-a9afd200-457e-11eb-96a1-03c1b28e08f9.png)

![image](https://user-images.githubusercontent.com/62425457/103028965-cc41eb00-457e-11eb-8237-28615ef08efe.png)

### Evolution of Results over epochs

![ezgif com-gif-maker](https://user-images.githubusercontent.com/62425457/103030298-88041a00-4581-11eb-95c1-c0d0b93eedad.gif)

 - From Omkar with love 💙 !!!
