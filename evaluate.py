import matplotlib.pyplot as plt
import numpy as np
import os, glob
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
np.random.seed(20180621)

from skimage.external import tifffile # read tiff images
from skimage.io import imread # read gif images
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
import seaborn as sns

from model import *
import utils
import time

#import cv2
plt.style.use('ggplot')

n_original_train = 100
n_original_test = 6
imgsize = 256
n_channel = 3


# load test images
x_test = np.zeros((n_original_test,imgsize,imgsize,n_channel), dtype=np.uint8)
fns = sorted(glob.glob('./Patients_test/*.tif'))
x_test[:,:,:,0] = np.array([tifffile.imread(fn) for fn in fns])
fns = sorted(glob.glob('./PatientJD_test/*.tif'))
x_test[:,:,:,1] = np.array([tifffile.imread(fn) for fn in fns])
fns = sorted(glob.glob('./PatientCV_test/*.tif'))
x_test[:,:,:,2]= np.array([tifffile.imread(fn) for fn in fns])
print('shape of raw test data: ', x_test.shape)

# load test annotations
fns = sorted(glob.glob('./label_test/*.tif'))
y_test = np.array([tifffile.imread(fn) for fn in fns]) # read images
y_test = np.expand_dims(y_test, -1) # add channels dimension
print('test shape:', y_test.shape)

x_test = utils.preprocess(x_test)
y_test = utils.preprocess(y_test)

# In[7]:


model = UNet(N_filters=32)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy', dice_coefficient, precision_smooth, recall_smooth])
print("Number of parameters: ", model.count_params())
pretrained_weights='./weights/sample_low.hdf5'

from keras.models import load_model
model.load_weights(pretrained_weights)


num_stochastic_T = 30
for i in range(6):
    image = x_test[i]
    gt    = y_test[i]
    prediction, aleatoric, epistemic, scores = utils.predict(model, image, gt, T=num_stochastic_T)
    x_test[i,:,:,1] = prediction*255
    x_test[i,:,:,2] = (epistemic + aleatoric)*10*255
    
plt.rcParams['figure.figsize'] = [20, 5]
M=6
list_index = np.random.randint(low=0, high=x_test.shape[0], size=M)
plt.figure(figsize=(25,25))
fig, ax = plt.subplots(4,M)  
for k, ind in enumerate(list_index):
    ax[0,k].imshow(x_test[ind,:,:,0], cmap='gray')
    ax[1,k].imshow(y_test[ind,:,:,0], cmap='gray')
    ax[2,k].imshow(x_test[ind,:,:,1], cmap='gray')
    ax[3,k].imshow(x_test[ind,:,:,2]) 
    ax[0,k].axis('off')
    ax[1,k].axis('off')
    ax[2,k].axis('off')
    ax[3,k].axis('off')
    
fig.savefig('./fig/testing_outputlow.pdf')

#
#image = x_test[1]
#gt    = y_test[1]
#prediction, aleatoric, epistemic, scores = utils.predict(model, image, gt, T=num_stochastic_T)
#    