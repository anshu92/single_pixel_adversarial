'''
Parts of code based on: https://github.com/Hyperparticle/one-pixel-attack-keras/blob/master/one-pixel-attack.ipynb
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import os
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras import backend as K
import imageio
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from skimage.transform import resize

'''
Code below is modified from https://github.com/Hyperparticle/one-pixel-attack-keras/blob/master/one-pixel-attack.ipynb
'''
def perturb_image(x, img):
    # Copy the image to keep the original unchanged
    img = np.copy(img) 
    
    # Split into an array of 3-tuples (perturbation pixels)
    # Make sure x to has its members of type int
    pixels = np.split(x.astype(int), len(x) // 3)
    
    # At each pixel's x,y position, assign its grayscale value
    for pixel in pixels:
        x_pos, y_pos, bw = pixel
        img[x_pos, y_pos,:] = bw
    return img

def predict_class(x, img, target_class, model, minimize=True):
    # Perturb the image with the given pixel(s) x and get the prediction of the model
    img_perturbed = perturb_image(x, img)
    img_perturbed = np.expand_dims(img_perturbed, axis=0)
    # img_perturbed = np.expand_dims(img_perturbed, axis=3)
    prediction = model.predict(img_perturbed)[0][target_class]
    # print(prediction)    
    # This function should always be minimized, so return its complement if needed
    return prediction if minimize else 1 - prediction

def attack_success(x, img, target_class, model, targeted_attack=False, verbose=False):
    # Perturb the image with the given pixel(s) and get the prediction of the model
    attack_image = perturb_image(x, img)
    attack_image = np.expand_dims(attack_image, axis=0)
    # attack_image = np.expand_dims(attack_image, axis=3)
    confidence = model.predict(attack_image)
    predicted_class = np.argmax(confidence[0])
    
    # If the prediction is what we want (misclassification or 
    # targeted classification), return True
    if (verbose):
        print('Confidence:', confidence[0][target_class])
    if ((targeted_attack and predicted_class == target_class) or
        (not targeted_attack and predicted_class != target_class)):
        return True
'''
Code below this contains functions that apply the adversarial attack logic to MNIST
'''

def get_ten_random(target_class):
    '''
    get 10 random images of given class
    '''
    # input image dimensions
    img_rows, img_cols = 28, 28
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    class2 = np.array([dat for dat, label in zip(x_train, y_train) if label == target_class])
    r = np.random.choice(class2.shape[0], 10)
    if not os.path.isdir(str(target_class)): os.mkdir(str(target_class))
    newdata = []
    for ix in r:
        im_arr = class2[ix,:,:,:]
        imageio.imwrite(str(target_class) + "/" + str(ix) + ".png", np.squeeze(im_arr))
        print(str(target_class) + "/" + str(ix) + ".png")
        newdata.append(im_arr)
    newdata = np.array(newdata)
    return newdata



# Our application logic will be added here
def adversarial_targeted():
    newdata = get_ten_random(2) # gets ten random images of class 2 
    target_class = 6 # class to target the examples towards
    bounds = [(0,28), (0,28), (0,1.)] # this sets the limits for width, height and pixel values
    
    model_ = load_model('mnist_model.hdf5') #load trained model: see retrain_inception_keras.py
    model = Sequential()
    model.add(model_)
    
    cnt = 1
    for img in newdata:

        ## function that differential evolution algorithm will optimize
        predict_fn = lambda x: predict_class(
            x, img, target_class, model, True) # x is the variable we wanna optimize, it represents (x, y, pixel value)
        ## function that differential evolution algorithm will callback to verify that the target class has been achieved
        callback_fn = lambda x, convergence: attack_success(
            x, img, target_class, model, True, False)
        ## this is scipy's implementation of differential evolution
        attack_result = differential_evolution(
            predict_fn, bounds, maxiter=30, popsize=30,
            recombination=1, atol=-1, callback=callback_fn)
        ## get the adversarial image
        attack_image = perturb_image(attack_result.x, img)

        ## plot original image
        plt.subplot(10,3,cnt)
        orig_im = np.squeeze(img)*255.
        plt.imshow(orig_im, cmap='gray')
        img_tensor = np.expand_dims(img, axis=0)
        pred_logit = np.max(model.predict(img_tensor)[0])
        plt.title('class 2 confidence: ' + "%.2f" % (pred_logit), fontdict={'fontsize':8})
        plt.xticks([])
        plt.yticks([])

        ## plot adversarial image
        plt.subplot(10,3,cnt+2)
        attack_im = np.squeeze(attack_image)*255.
        plt.imshow(attack_im, cmap='gray')
        img_tensor = np.expand_dims(attack_image, axis=0)
        pred_logit = np.max(model.predict(img_tensor)[0])
        plt.title('class 6 confidence: ' + "%.2f" % (pred_logit), fontdict={'fontsize':8})
        plt.xticks([])
        plt.yticks([])

        ## plot difference between the two images
        plt.subplot(10,3,cnt+1)
        diff_im = np.abs(np.squeeze(attack_image-img))*255.
        plt.imshow(diff_im, cmap='gray')
        indices = np.where(diff_im != 0)
        x, y = indices[0][0], indices[1][0] # location of changed pixel
        plt.title("x,y: " + str((x,y)) + " orignal: " + str(int(orig_im[x,y])) + " adversarial: " + str(int(attack_im[x,y])), fontdict={'fontsize':8})
        plt.xticks([])
        plt.yticks([])
        cnt += 3
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.25)
    plt.savefig('output_debug.png')
    plt.show()

if __name__ == "__main__":
  adversarial_targeted()