import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Input, MaxPool2D, UpSampling2D, Concatenate, Conv2DTranspose
import tensorflow as tf
from keras.optimizers import Adam
from scipy.misc import imresize
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import sys
from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
#%matplotlib inline
from subprocess import check_output
print(check_output(["ls", "input"]).decode("utf8"))
from PIL import Image
from keras.models import model_from_yaml

data_dir = "input/train/train/"
mask_dir = "input/train_masks/"
#test_dir="input/test/"
all_images = os.listdir(data_dir)

#all_test_images=os.listdir(test_dir)

train_images, validation_images = all_images[:3000],all_images[3000:]#train_test_split(all_images, train_size=0.8, test_size=0.2)

def grey2rgb(img):
    new_img = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img.append(list(img[i][j])*3)
    new_img = np.array(new_img).reshape(img.shape[0], img.shape[1], 3)
    return new_img


def rle (img):
    flat_img = img.flatten()

    flat_img = np.where(flat_img > 0.5, 1, 0).astype(np.uint8)
    flat_img = np.insert(flat_img, [0, len(flat_img)], [0, 0])
    
    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))

    starts_ix = np.where(starts)[0] + 2    
    ends_ix = np.where(ends)[0] + 2
    
    lengths = ends_ix - starts_ix
    #print lengths  

    encoding = ''
    for idx in range(len(starts_ix)):
        encoding += '%d %d ' % (starts_ix[idx], lengths[idx])
    return encoding
    #return starts_ix, lengths

# generator that we will use to read the data from the directory
def data_gen(data_dir, mask_dir, images, batch_size, dims):
        """
        data_dir: where the actual images are kept
        mask_dir: where the actual masks are kept
        images: the filenames of the images we want to generate batches from
        batch_size: self explanatory
        dims: the dimensions in which we want to rescale our images
        """
        while True:
            ix = np.random.choice(np.arange(len(images)), batch_size)
            imgs = []
            labels = []
            for i in ix:
                # images
                original_img = load_img(data_dir + images[i])
               
		resized_img = imresize(original_img, dims+[3])  
		array_img = img_to_array(resized_img)/255
                imgs.append(array_img)
                
                # masks
                original_mask = load_img(mask_dir + images[i].split(".")[0] + '_mask.gif')
                resized_mask = imresize(original_mask, dims+[3])
                array_mask = img_to_array(resized_mask)/255             
		labels.append(array_mask[:, :, 0])
            imgs = np.array(imgs)
            labels = np.array(labels)
	    #print labels.reshape(-1, dims[0], dims[1], 1).shape
	
            yield imgs, labels.reshape(-1, dims[0], dims[1], 1)


def test_generator(data_dir, images, start,finish, dims):
           
	    #ix = np.random.choice(np.arange(len(images)), batch_size)
            imgs = []
            filenames = []
            for i in range(start,finish):
                # images
                original_img = load_img(data_dir + images[i])
               
		resized_img = imresize(original_img, dims+[3])  
		array_img = img_to_array(resized_img)/255
                imgs.append(array_img)
		filenames.append(images[i])
                
                # masks
                '''original_mask = load_img(mask_dir + images[i].split(".")[0] + '_mask.gif')
                resized_mask = imresize(original_mask, dims+[3])
                array_mask = img_to_array(resized_mask)/255             
		labels.append(array_mask[:, :, 0])'''
            imgs = np.array(imgs)
	    filename=np.array(filename)
            #labels = np.array(labels)
	    #print labels.reshape(-1, dims[0], dims[1], 1).shape
	
            return imgs, filenames
	    




#plt.imshow(img[0])
#plt.imshow(grey2rgb(msk[0]), alpha=0.5)





def down(input_layer, filters, pool=True):
    conv1 = Conv2D(filters, (3, 3), padding='same', activation='relu')(input_layer)
    residual = Conv2D(filters, (3, 3), padding='same', activation='relu')(conv1)
    if pool:
        max_pool = MaxPool2D()(residual)
        return max_pool, residual
    else:
        return residual

def up(input_layer, residual, filters):
    filters=int(filters)
    upsample = UpSampling2D()(input_layer)
    upconv = Conv2D(filters, kernel_size=(2, 2), padding="same")(upsample)
    concat = Concatenate(axis=3)([residual, upconv])
    conv1 = Conv2D(filters, (3, 3), padding='same', activation='relu')(concat)
    conv2 = Conv2D(filters, (3, 3), padding='same', activation='relu')(conv1)
    return conv2





# Make a custom U-nets implementation.
filters = 64
input_layer = Input(shape = [128, 128, 3])
layers = [input_layer]
residuals = []

# Down 1, 128
d1, res1 = down(input_layer, filters)
residuals.append(res1)

filters *= 2

# Down 2, 64
d2, res2 = down(d1, filters)
residuals.append(res2)

filters *= 2

# Down 3, 32
d3, res3 = down(d2, filters)
residuals.append(res3)

filters *= 2

# Down 4, 16
d4, res4 = down(d3, filters)
residuals.append(res4)

filters *= 2

# Down 5, 8
d5 = down(d4, filters, pool=False)

# Up 1, 16
up1 = up(d5, residual=residuals[-1], filters=filters/2)

filters /= 2

# Up 2,  32
up2 = up(up1, residual=residuals[-2], filters=filters/2)

filters /= 2

# Up 3, 64
up3 = up(up2, residual=residuals[-3], filters=filters/2)

filters /= 2

# Up 4, 128
up4 = up(up3, residual=residuals[-4], filters=filters/2)
out = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid")(up4)
model = Model(input_layer, out)
#model.summary()


# Now let's use Tensorflow to write our own dice_coeficcient metric
def dice_coef(y_true, y_pred):
    smooth = 1e-5
    
    y_true = tf.round(tf.reshape(y_true, [-1]))
    y_pred = tf.round(tf.reshape(y_pred, [-1]))
    
    isct = tf.reduce_sum(y_true * y_pred)
    
    return 2 * isct / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))


# example use
batch_size=5
train_gen = data_gen(data_dir, mask_dir, train_images, batch_size, [128,128])#[1918, 1280])
val_gen = data_gen(data_dir, mask_dir, validation_images, batch_size, [128,128])#[1918, 1280])
img, msk = next(train_gen)
img_val, msk_val = next(val_gen)



model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=[dice_coef])
model.fit_generator(train_gen, steps_per_epoch=50, epochs=1, validation_data=val_gen, validation_steps=150, verbose=1)

model_yaml = model.to_yaml()
with open("carvana.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
model.save_weights("weights.h5")
print("Saved model to disk")


#load model for testing
yaml_file = open('carvana.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)

print("Loaded model and weights from disk")
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=[dice_coef])
loaded_model.load_weights("weights.h5")

def test_generator(data_dir, images, start,finish, dims):
           
	    #ix = np.random.choice(np.arange(len(images)), batch_size)
            imgs = []
            filenames = []
            for i in range(start,finish):
                # images
                original_img = load_img(data_dir + images[i])
               
		resized_img = imresize(original_img, dims+[3])  
		array_img = img_to_array(resized_img)/255
                imgs.append(array_img)
		filenames.append(images[i])
                
                # masks
                '''original_mask = load_img(mask_dir + images[i].split(".")[0] + '_mask.gif')
                resized_mask = imresize(original_mask, dims+[3])
                array_mask = img_to_array(resized_mask)/255             
		labels.append(array_mask[:, :, 0])'''
            imgs = np.array(imgs)
	    filenames=np.array(filenames)
            #labels = np.array(labels)
	    #print labels.reshape(-1, dims[0], dims[1], 1).shape
	
            return imgs, filenames


test_gen=test_generator(data_dir, validation_images, 0,5, [128, 128])
img,filename=next(test_gen)
img_pred = test_generator(data_dir, validation_images, batch_size, [128, 128])


#predict and convert to rle to write to file
for i in range(len(validation_images)):

	imgs,filenames=test_generator(data_dir, validation_images, 0,5, [128, 128])
	#print imgs
	preds=model.predict(imgs, batch_size=batch_size,verbose=0)
	for i in preds:
		x=rle(i.T[0])




