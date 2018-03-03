# single_pixel_adversarial
Implementation of https://arxiv.org/abs/1710.08864 on MNIST

## requirements:
 - keras
 - tensorflow
 - matplotlib

## Training MNIST classifier:
 - example code taken from keras examples - https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

## Targeted adversarial example generation:
- parts of code taken from https://github.com/Hyperparticle/one-pixel-attack-keras/blob/master/one-pixel-attack.ipynb

### Differential evolution
- This is the optimization algorithm used to find the parameters we are looking for - location of pixel (x,y) and value of pixel. It is a population based black box algorithm that generates new candidates at each iteration by combining best performing candidates in the current selection. The method does not use gradients but rather moving towards the global minima by following a linear combination of best performing 'fathers'. In this implementation I follow the structure of the code above, and use scipy implementation of the algorithm.

### parameter functions
- predict_class function: takes the generated parameters as input and returns the prediction value of image with changed single pixel. This informs the DE algorithm of candidate performance.

- attack_success function: tests whether the candidate changes the original class to target class (2 to 6). DE uses it to calculate whether minima is reached.

Finally, I got ten random images of class 2 and used the algorithm to find adversarial examples that cause the model to misclassify the class to 6.

![example_image](https://github.com/anshu92/single_pixel_adversarial/blob/master/output_debug.png "Example Image")

