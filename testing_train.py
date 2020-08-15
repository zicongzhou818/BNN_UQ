import matplotlib.pyplot as plt
import numpy as np
import os, glob
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
np.random.seed(20180621)

#import cv2

from skimage.external import tifffile # read tiff images
from skimage.io import imread # read gif images
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
import seaborn as sns

from array import *
from model import *
import utils
import time

plt.style.use('ggplot')

print('Backend: ', K.backend())
print('Image_data_format: ', K.image_data_format())
N_train = 500
N_test = 100
n_original_train = 100
n_original_test = 6
samplingsize = 192
imgsize = 256
n_channel = 3
x_train = np.zeros((n_original_train,imgsize,imgsize,n_channel), dtype=np.uint8)
fns = sorted(glob.glob('./Patients/*.tif'))
x_train[:,:,:,0] = np.array([tifffile.imread(fn) for fn in fns])
fns = sorted(glob.glob('./PatientJD/*.tif'))
x_train[:,:,:,1] = np.array([tifffile.imread(fn) for fn in fns])
fns = sorted(glob.glob('./PatientCV/*.tif'))
x_train[:,:,:,2] = np.array([tifffile.imread(fn) for fn in fns])
print('shape of raw train data: ', x_train.shape)

# load test images
x_test = np.zeros((n_original_test,imgsize,imgsize,n_channel), dtype=np.uint8)
fns = sorted(glob.glob('./Patients_test/*.tif'))
x_test[:,:,:,0] = np.array([tifffile.imread(fn) for fn in fns])
fns = sorted(glob.glob('./PatientJD_test/*.tif'))
x_test[:,:,:,1] = np.array([tifffile.imread(fn) for fn in fns])
fns = sorted(glob.glob('./PatientCV_test/*.tif'))
x_test[:,:,:,2]= np.array([tifffile.imread(fn) for fn in fns])
print('shape of raw test data: ', x_test.shape)

# load training annotations
fns = sorted(glob.glob('./label/*.tif'))
y_train = np.array([tifffile.imread(fn) for fn in fns]) # read images
y_train = np.expand_dims(y_train, -1) # add channels dimension
print('train shape:', y_train.shape)

# load test annotations
fns = sorted(glob.glob('./label_test/*.tif'))
y_test = np.array([tifffile.imread(fn) for fn in fns]) # read images
y_test = np.expand_dims(y_test, -1) # add channels dimension
print('test shape:', y_test.shape)


# preprocessing
x_train = utils.preprocess(x_train)
x_test = utils.preprocess(x_test)

y_train = utils.preprocess(y_train)
y_test = utils.preprocess(y_test)

X_train, Y_train = utils.get_random_snippets(x_train, y_train, number=N_train, size=(samplingsize,samplingsize))
X_test, Y_test = utils.get_random_snippets(x_test, y_test, number=N_test, size=(samplingsize,samplingsize))



print('X_train shape: ', X_train.shape)
print('Y_train shape: ', Y_train.shape)
print('X_test shape: ', X_test.shape)
print('Y_test shape: ', Y_test.shape)

plt.rcParams['figure.figsize'] = [25, 5]
M=6
list_index = np.random.randint(low=0, high=X_train.shape[0], size=M)
plt.figure(figsize=(25,25))
fig, ax = plt.subplots(4,M)
for k, ind in enumerate(list_index):
    ax[0,k].imshow(X_train[ind,:,:,0], cmap='gray')
    ax[1,k].imshow(X_train[ind,:,:,1], cmap='gray')
    ax[2,k].imshow(X_train[ind,:,:,2], cmap='gray')
    ax[3,k].imshow(Y_train[ind,:,:,0], cmap='gray')
    
    ax[0,k].axis('off')
    ax[1,k].axis('off')
    ax[2,k].axis('off')
    ax[3,k].axis('off')

fig.savefig('./fig/example_high.pdf')

model = UNet(N_filters=32)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy', dice_coefficient, precision_smooth, recall_smooth])
print("Number of parameters: ", model.count_params())

batch_size = 2
epochs = 500
info_check_string='./weights/sample_high.hdf5'
early_stopping=EarlyStopping(monitor='val_loss', patience=200)
model_checkpoint=ModelCheckpoint(info_check_string, monitor='loss', save_best_only=True)

history = model.fit(X_train, Y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      shuffle=True,
                      verbose=1,
                      validation_split=0.1, # 4 samples are used for a validation set
                      callbacks=[early_stopping, model_checkpoint])

score = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss, acc, dice, precision, recall):', score)

def plot_history(history, validation=False):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5), sharex=True)
    #fig.tight_layout()
    # plot history for loss
    ax.plot(history.history['loss'])
    if validation:
        ax.plot(history.history['val_loss'])
    ax.set_title('Loss')
    ax.set_ylabel('Loss')
    ax.set_ylim(bottom=0.)
    ax.set_xlabel('Epoch')
    ax.legend(['train', 'test'])
    
    plt.savefig('./fig/loss_curve_high.pdf')
    
plot_history(history, validation=True)

# It takes a time !! 
num = len(X_test)
list_stochastic_feed_forwards = [3,5,10,15,20,25,30]
result_dict = {}
for ind, num_stochastic_T in enumerate(list_stochastic_feed_forwards):
    start = time.time()
    alea_list = []
    epis_list = []
    dice_list = []
    for i in range(num):
        image = X_test[i]
        gt    = Y_test[i]
        prediction, aleatoric, epistemic, scores = utils.predict(model, image, gt, T=num_stochastic_T)
        alea_list.append(np.mean(aleatoric))
        epis_list.append(np.mean(epistemic))
        dice_list.append(scores[0])
    
    end = time.time()
    print('-'*30)
    print('T: ', num_stochastic_T)
    print('Exec time per prediction = {:.3f}'.format((end-start)/num))
    print('aleatoric: ', np.mean(alea_list), np.std(alea_list))
    print('epistemic: ', np.mean(epis_list), np.std(epis_list))    
    print('Dice: ', np.mean(dice_list), np.std(dice_list))  
    print('-'*30)
    
    result_dict.update({ '{}'.format(str(num_stochastic_T)) : 
    [num_stochastic_T, (end-start)/num,
     np.mean(alea_list), np.std(alea_list),
     np.mean(epis_list), np.std(epis_list),
     np.mean(dice_list), np.std(dice_list)]} )
    
results = np.zeros((5, len(list_stochastic_feed_forwards)))    
for ind, num_stochastic_T in enumerate(list_stochastic_feed_forwards):
    results[:, ind] = result_dict[str(num_stochastic_T)][1:6]
    
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(8,10), sharex=True)
# plot history for loss
ax[0].scatter(list_stochastic_feed_forwards,results[0])
ax[0].set_ylabel('Elapsed time per subject')
ax[1].errorbar(list_stochastic_feed_forwards,results[1], yerr=results[2]/np.sqrt(list_stochastic_feed_forwards))
ax[1].set_ylabel('Aleatoric uncertainty')
ax[2].errorbar(list_stochastic_feed_forwards,results[3], yerr=results[4]/np.sqrt(list_stochastic_feed_forwards))
ax[2].set_ylabel('Epistemic uncertainty')
ax[2].set_xlabel('Number of realized vectors ($T$)')
ax[2].set_ylim(0.00,0.01)
plt.savefig('./fig/elapsed_time_vs_feed_high.pdf')

# Case T = 5
num = len(X_test)
start = time.time()
list_alea = []
list_epis = []
for i in range(num):
    image = X_test[i]
    gt    = Y_test[i]
    prediction, aleatoric, epistemic, scores = utils.predict(model, image, gt, T=5)
    list_alea.append(aleatoric.reshape(-1))
    list_epis.append(epistemic.reshape(-1))

end = time.time()
print('Exec time per prediction = {:.3f}'.format((end-start)/num))
print('aleatoric: ', np.mean(list_alea))
print('epistemic: ', np.mean(list_epis))

list_alea = np.hstack(list_alea)
list_epis = np.hstack(list_epis)

print('aleatoric mean: ', np.mean(list_alea))
print('epistemic mean: ', np.mean(list_epis))

threshold = np.percentile(list_alea, 99)
alea_index = (list_alea > threshold)

plt.figure(figsize=(6,6))
data = np.vstack([list_alea[alea_index], list_epis[alea_index]]).T
ax = sns.kdeplot(data, shade = True, cmap = "gray", cbar=False)
ax.patch.set_facecolor('white')
ax.collections[0].set_alpha(0)
ax.set_xlabel('Aleatoric', fontsize = 15)
ax.set_ylabel('Epistemic', fontsize = 15)
ax.set_xlim(0.15, 0.25)
ax.set_ylim(0, 0.1)
plt.savefig('./fig/epis_vs_alea_low.pdf')

threshold_99 = np.percentile(list_alea, 99.)
threshold_995 = np.percentile(list_alea, 99.5)
threshold_999 = np.percentile(list_alea, 99.9)
threshold_9995 = np.percentile(list_alea, 99.95)
print(threshold_99, threshold_995, threshold_999, threshold_9995)
# alea_index = (list_alea > threshold)

# Table 5 in the paper.
print('conditional expectation of epistemic')
print(
    np.mean([list_epis[i] for i in np.arange(len(list_alea)) if (list_alea[i] > 0.05) and (list_alea[i] < 0.1)]),
    np.mean([list_epis[i] for i in np.arange(len(list_alea)) if (list_alea[i] > 0.1) and (list_alea[i] < 0.15)]),
    np.mean([list_epis[i] for i in np.arange(len(list_alea)) if (list_alea[i] > 0.15) and (list_alea[i] < 0.2)]),
    np.mean([list_epis[i] for i in np.arange(len(list_alea)) if (list_alea[i] > 0.2) and (list_alea[i] < 0.25)])
    )


#############################################################
#pretrained_weights='./weights/sample.hdf5'
#Using TensorFlow backend.
#Backend:  tensorflow
#Image_data_format:  channels_last
#shape of raw train data:  (100, 256, 256, 3)
#shape of raw test data:  (6, 256, 256, 3)
#train shape: (100, 256, 256, 1)
#test shape: (6, 256, 256, 1)
#min: 0.0, max: 1.0, shape: (100, 256, 256, 3), type: float32
#min: 0.0, max: 1.0, shape: (6, 256, 256, 3), type: float32
#min: 0.0, max: 1.0, shape: (100, 256, 256, 1), type: float32
#min: 0.0, max: 1.0, shape: (6, 256, 256, 1), type: float32
#X_train shape:  (1000, 192, 192, 3)
#Y_train shape:  (1000, 192, 192, 1)
#X_test shape:  (100, 192, 192, 3)
#Y_test shape:  (100, 192, 192, 1)
#Number of parameters:  1115265
#Train on 800 samples, validate on 200 samples
#Epoch 1/200
#800/800 [==============================] - 321s 402ms/step - loss: 0.7270 - binary_accuracy: 0.5651 - dice_coefficient: 0.0947 - precision_smooth: 0.0520 - recall_smooth: 0.5485 - val_loss: 0.7106 - val_binary_accuracy: 0.5553 - val_dice_coefficient: 0.1002 - val_precision_smooth: 0.0551 - val_recall_smooth: 0.5671
#Epoch 2/200
#800/800 [==============================] - 320s 400ms/step - loss: 0.5476 - binary_accuracy: 0.7959 - dice_coefficient: 0.1014 - precision_smooth: 0.0566 - recall_smooth: 0.5030 - val_loss: 0.5947 - val_binary_accuracy: 0.7146 - val_dice_coefficient: 0.0869 - val_precision_smooth: 0.0485 - val_recall_smooth: 0.4269
#Epoch 3/200
#800/800 [==============================] - 347s 434ms/step - loss: 0.4071 - binary_accuracy: 0.9137 - dice_coefficient: 0.1076 - precision_smooth: 0.0619 - recall_smooth: 0.4308 - val_loss: 0.3947 - val_binary_accuracy: 0.8812 - val_dice_coefficient: 0.0778 - val_precision_smooth: 0.0459 - val_recall_smooth: 0.2629
#Epoch 4/200
#800/800 [==============================] - 355s 444ms/step - loss: 0.3060 - binary_accuracy: 0.9350 - dice_coefficient: 0.1108 - precision_smooth: 0.0664 - recall_smooth: 0.3448 - val_loss: 0.3042 - val_binary_accuracy: 0.9214 - val_dice_coefficient: 0.0997 - val_precision_smooth: 0.0606 - val_recall_smooth: 0.2880
#Epoch 5/200
#800/800 [==============================] - 347s 434ms/step - loss: 0.2460 - binary_accuracy: 0.9391 - dice_coefficient: 0.1056 - precision_smooth: 0.0670 - recall_smooth: 0.2574 - val_loss: 0.5673 - val_binary_accuracy: 0.7290 - val_dice_coefficient: 0.0855 - val_precision_smooth: 0.0495 - val_recall_smooth: 0.3212
#Epoch 6/200
#800/800 [==============================] - 346s 432ms/step - loss: 0.2083 - binary_accuracy: 0.9421 - dice_coefficient: 0.1059 - precision_smooth: 0.0722 - recall_smooth: 0.2061 - val_loss: 0.2905 - val_binary_accuracy: 0.9013 - val_dice_coefficient: 0.0678 - val_precision_smooth: 0.0465 - val_recall_smooth: 0.1272
#Epoch 7/200
#800/800 [==============================] - 380s 475ms/step - loss: 0.1887 - binary_accuracy: 0.9391 - dice_coefficient: 0.1203 - precision_smooth: 0.0862 - recall_smooth: 0.2048 - val_loss: 0.2126 - val_binary_accuracy: 0.9359 - val_dice_coefficient: 0.0636 - val_precision_smooth: 0.0519 - val_recall_smooth: 0.0835
#Epoch 8/200
#800/800 [==============================] - 343s 428ms/step - loss: 0.1692 - binary_accuracy: 0.9412 - dice_coefficient: 0.1369 - precision_smooth: 0.1037 - recall_smooth: 0.2053 - val_loss: 0.1814 - val_binary_accuracy: 0.9410 - val_dice_coefficient: 0.0810 - val_precision_smooth: 0.0666 - val_recall_smooth: 0.1049
#Epoch 9/200
#800/800 [==============================] - 346s 433ms/step - loss: 0.1564 - binary_accuracy: 0.9420 - dice_coefficient: 0.1569 - precision_smooth: 0.1246 - recall_smooth: 0.2155 - val_loss: 0.1638 - val_binary_accuracy: 0.9408 - val_dice_coefficient: 0.1080 - val_precision_smooth: 0.0945 - val_recall_smooth: 0.1278
#Epoch 10/200
#800/800 [==============================] - 379s 473ms/step - loss: 0.1491 - binary_accuracy: 0.9424 - dice_coefficient: 0.1740 - precision_smooth: 0.1430 - recall_smooth: 0.2278 - val_loss: 0.1610 - val_binary_accuracy: 0.9400 - val_dice_coefficient: 0.1163 - val_precision_smooth: 0.1100 - val_recall_smooth: 0.1254
#Epoch 11/200
#800/800 [==============================] - 379s 473ms/step - loss: 0.1435 - binary_accuracy: 0.9425 - dice_coefficient: 0.1831 - precision_smooth: 0.1557 - recall_smooth: 0.2259 - val_loss: 0.1527 - val_binary_accuracy: 0.9420 - val_dice_coefficient: 0.1285 - val_precision_smooth: 0.1332 - val_recall_smooth: 0.1253
#Epoch 12/200
#800/800 [==============================] - 355s 444ms/step - loss: 0.1383 - binary_accuracy: 0.9428 - dice_coefficient: 0.1965 - precision_smooth: 0.1717 - recall_smooth: 0.2368 - val_loss: 0.1497 - val_binary_accuracy: 0.9406 - val_dice_coefficient: 0.1801 - val_precision_smooth: 0.1517 - val_recall_smooth: 0.2247
#Epoch 13/200
#800/800 [==============================] - 340s 424ms/step - loss: 0.1363 - binary_accuracy: 0.9429 - dice_coefficient: 0.2002 - precision_smooth: 0.1794 - recall_smooth: 0.2325 - val_loss: 0.1789 - val_binary_accuracy: 0.9346 - val_dice_coefficient: 0.1484 - val_precision_smooth: 0.1092 - val_recall_smooth: 0.2365
#Epoch 14/200
#800/800 [==============================] - 345s 431ms/step - loss: 0.1337 - binary_accuracy: 0.9436 - dice_coefficient: 0.2156 - precision_smooth: 0.1940 - recall_smooth: 0.2477 - val_loss: 0.1553 - val_binary_accuracy: 0.9394 - val_dice_coefficient: 0.1392 - val_precision_smooth: 0.1512 - val_recall_smooth: 0.1300
#Epoch 15/200
#800/800 [==============================] - 341s 426ms/step - loss: 0.1294 - binary_accuracy: 0.9438 - dice_coefficient: 0.2305 - precision_smooth: 0.2088 - recall_smooth: 0.2621 - val_loss: 0.1453 - val_binary_accuracy: 0.9394 - val_dice_coefficient: 0.1718 - val_precision_smooth: 0.1664 - val_recall_smooth: 0.1808
#Epoch 16/200
#800/800 [==============================] - 330s 412ms/step - loss: 0.1289 - binary_accuracy: 0.9434 - dice_coefficient: 0.2325 - precision_smooth: 0.2149 - recall_smooth: 0.2616 - val_loss: 0.1709 - val_binary_accuracy: 0.9420 - val_dice_coefficient: 0.1012 - val_precision_smooth: 0.1588 - val_recall_smooth: 0.0747
#Epoch 17/200
#800/800 [==============================] - 351s 439ms/step - loss: 0.1251 - binary_accuracy: 0.9446 - dice_coefficient: 0.2440 - precision_smooth: 0.2291 - recall_smooth: 0.2681 - val_loss: 0.1685 - val_binary_accuracy: 0.9404 - val_dice_coefficient: 0.1192 - val_precision_smooth: 0.0945 - val_recall_smooth: 0.1638
#Epoch 18/200
#800/800 [==============================] - 340s 425ms/step - loss: 0.1225 - binary_accuracy: 0.9445 - dice_coefficient: 0.2507 - precision_smooth: 0.2375 - recall_smooth: 0.2724 - val_loss: 0.1911 - val_binary_accuracy: 0.9129 - val_dice_coefficient: 0.2293 - val_precision_smooth: 0.1600 - val_recall_smooth: 0.4146
#Epoch 19/200
#800/800 [==============================] - 338s 423ms/step - loss: 0.1178 - binary_accuracy: 0.9459 - dice_coefficient: 0.2810 - precision_smooth: 0.2626 - recall_smooth: 0.3085 - val_loss: 0.1847 - val_binary_accuracy: 0.9249 - val_dice_coefficient: 0.1541 - val_precision_smooth: 0.1142 - val_recall_smooth: 0.2418
#Epoch 20/200
#800/800 [==============================] - 350s 438ms/step - loss: 0.1165 - binary_accuracy: 0.9466 - dice_coefficient: 0.2861 - precision_smooth: 0.2727 - recall_smooth: 0.3076 - val_loss: 0.2229 - val_binary_accuracy: 0.8971 - val_dice_coefficient: 0.2321 - val_precision_smooth: 0.1523 - val_recall_smooth: 0.4989
#Epoch 21/200
#800/800 [==============================] - 341s 426ms/step - loss: 0.1164 - binary_accuracy: 0.9459 - dice_coefficient: 0.2839 - precision_smooth: 0.2701 - recall_smooth: 0.3096 - val_loss: 0.1525 - val_binary_accuracy: 0.9422 - val_dice_coefficient: 0.1583 - val_precision_smooth: 0.2531 - val_recall_smooth: 0.1159
#Epoch 22/200
#800/800 [==============================] - 354s 442ms/step - loss: 0.1095 - binary_accuracy: 0.9480 - dice_coefficient: 0.3220 - precision_smooth: 0.3060 - recall_smooth: 0.3493 - val_loss: 0.2142 - val_binary_accuracy: 0.9008 - val_dice_coefficient: 0.2324 - val_precision_smooth: 0.1538 - val_recall_smooth: 0.4841
#Epoch 23/200
#800/800 [==============================] - 337s 422ms/step - loss: 0.1400 - binary_accuracy: 0.9418 - dice_coefficient: 0.2210 - precision_smooth: 0.2218 - recall_smooth: 0.2348 - val_loss: 0.2895 - val_binary_accuracy: 0.9404 - val_dice_coefficient: 0.0119 - val_precision_smooth: 0.0492 - val_recall_smooth: 0.0068
#Epoch 24/200
#800/800 [==============================] - 335s 419ms/step - loss: 0.1182 - binary_accuracy: 0.9462 - dice_coefficient: 0.2785 - precision_smooth: 0.2705 - recall_smooth: 0.2913 - val_loss: 0.2506 - val_binary_accuracy: 0.9410 - val_dice_coefficient: 0.0165 - val_precision_smooth: 0.0758 - val_recall_smooth: 0.0093
#Epoch 25/200
#800/800 [==============================] - 371s 464ms/step - loss: 0.1132 - binary_accuracy: 0.9471 - dice_coefficient: 0.3060 - precision_smooth: 0.2953 - recall_smooth: 0.3254 - val_loss: 0.1743 - val_binary_accuracy: 0.9409 - val_dice_coefficient: 0.0972 - val_precision_smooth: 0.1678 - val_recall_smooth: 0.0690
#Epoch 26/200
#800/800 [==============================] - 349s 436ms/step - loss: 0.1113 - binary_accuracy: 0.9471 - dice_coefficient: 0.3118 - precision_smooth: 0.3012 - recall_smooth: 0.3325 - val_loss: 0.1550 - val_binary_accuracy: 0.9406 - val_dice_coefficient: 0.1862 - val_precision_smooth: 0.2860 - val_recall_smooth: 0.1395
#Epoch 27/200
#800/800 [==============================] - 363s 453ms/step - loss: 0.1076 - binary_accuracy: 0.9483 - dice_coefficient: 0.3340 - precision_smooth: 0.3230 - recall_smooth: 0.3505 - val_loss: 0.2604 - val_binary_accuracy: 0.8753 - val_dice_coefficient: 0.2056 - val_precision_smooth: 0.1351 - val_recall_smooth: 0.4427
#Epoch 28/200
#800/800 [==============================] - 353s 441ms/step - loss: 0.1053 - binary_accuracy: 0.9494 - dice_coefficient: 0.3503 - precision_smooth: 0.3393 - recall_smooth: 0.3692 - val_loss: 0.1700 - val_binary_accuracy: 0.9300 - val_dice_coefficient: 0.1734 - val_precision_smooth: 0.1686 - val_recall_smooth: 0.1831
#Epoch 29/200
#800/800 [==============================] - 348s 435ms/step - loss: 0.1002 - binary_accuracy: 0.9510 - dice_coefficient: 0.3780 - precision_smooth: 0.3642 - recall_smooth: 0.3995 - val_loss: 0.1847 - val_binary_accuracy: 0.9372 - val_dice_coefficient: 0.1407 - val_precision_smooth: 0.1811 - val_recall_smooth: 0.1179
#Epoch 30/200
#800/800 [==============================] - 357s 447ms/step - loss: 0.0997 - binary_accuracy: 0.9507 - dice_coefficient: 0.3785 - precision_smooth: 0.3712 - recall_smooth: 0.3954 - val_loss: 0.1504 - val_binary_accuracy: 0.9404 - val_dice_coefficient: 0.2209 - val_precision_smooth: 0.2588 - val_recall_smooth: 0.1952
#Epoch 31/200
#800/800 [==============================] - 350s 437ms/step - loss: 0.0978 - binary_accuracy: 0.9510 - dice_coefficient: 0.3907 - precision_smooth: 0.3801 - recall_smooth: 0.4116 - val_loss: 0.1589 - val_binary_accuracy: 0.9268 - val_dice_coefficient: 0.2760 - val_precision_smooth: 0.2205 - val_recall_smooth: 0.3764
#Epoch 32/200
#800/800 [==============================] - 355s 444ms/step - loss: 0.0968 - binary_accuracy: 0.9511 - dice_coefficient: 0.3969 - precision_smooth: 0.3891 - recall_smooth: 0.4149 - val_loss: 0.1299 - val_binary_accuracy: 0.9464 - val_dice_coefficient: 0.2644 - val_precision_smooth: 0.3808 - val_recall_smooth: 0.2034
#Epoch 33/200
#800/800 [==============================] - 341s 426ms/step - loss: 0.0920 - binary_accuracy: 0.9529 - dice_coefficient: 0.4225 - precision_smooth: 0.4117 - recall_smooth: 0.4456 - val_loss: 0.1368 - val_binary_accuracy: 0.9412 - val_dice_coefficient: 0.2433 - val_precision_smooth: 0.2466 - val_recall_smooth: 0.2441
#Epoch 34/200
#800/800 [==============================] - 340s 424ms/step - loss: 0.0892 - binary_accuracy: 0.9533 - dice_coefficient: 0.4323 - precision_smooth: 0.4215 - recall_smooth: 0.4530 - val_loss: 0.1111 - val_binary_accuracy: 0.9486 - val_dice_coefficient: 0.3850 - val_precision_smooth: 0.4163 - val_recall_smooth: 0.3624
#Epoch 35/200
#800/800 [==============================] - 338s 423ms/step - loss: 0.0869 - binary_accuracy: 0.9543 - dice_coefficient: 0.4548 - precision_smooth: 0.4462 - recall_smooth: 0.4731 - val_loss: 0.1287 - val_binary_accuracy: 0.9380 - val_dice_coefficient: 0.3723 - val_precision_smooth: 0.3123 - val_recall_smooth: 0.4666
#Epoch 36/200
#800/800 [==============================] - 331s 414ms/step - loss: 0.0847 - binary_accuracy: 0.9551 - dice_coefficient: 0.4688 - precision_smooth: 0.4572 - recall_smooth: 0.4900 - val_loss: 0.1193 - val_binary_accuracy: 0.9475 - val_dice_coefficient: 0.3303 - val_precision_smooth: 0.4094 - val_recall_smooth: 0.2802
#Epoch 37/200
#800/800 [==============================] - 339s 424ms/step - loss: 0.0825 - binary_accuracy: 0.9557 - dice_coefficient: 0.4782 - precision_smooth: 0.4673 - recall_smooth: 0.4967 - val_loss: 0.1333 - val_binary_accuracy: 0.9429 - val_dice_coefficient: 0.3191 - val_precision_smooth: 0.3216 - val_recall_smooth: 0.3214
#Epoch 38/200
#800/800 [==============================] - 334s 418ms/step - loss: 0.0818 - binary_accuracy: 0.9556 - dice_coefficient: 0.4848 - precision_smooth: 0.4751 - recall_smooth: 0.5067 - val_loss: 0.1436 - val_binary_accuracy: 0.9420 - val_dice_coefficient: 0.2669 - val_precision_smooth: 0.3605 - val_recall_smooth: 0.2143
#Epoch 39/200
#800/800 [==============================] - 349s 436ms/step - loss: 0.0810 - binary_accuracy: 0.9557 - dice_coefficient: 0.4858 - precision_smooth: 0.4860 - recall_smooth: 0.4967 - val_loss: 0.1582 - val_binary_accuracy: 0.9342 - val_dice_coefficient: 0.3062 - val_precision_smooth: 0.3040 - val_recall_smooth: 0.3132
#Epoch 40/200
#800/800 [==============================] - 351s 438ms/step - loss: 0.0756 - binary_accuracy: 0.9571 - dice_coefficient: 0.5197 - precision_smooth: 0.5058 - recall_smooth: 0.5427 - val_loss: 0.3524 - val_binary_accuracy: 0.8550 - val_dice_coefficient: 0.2516 - val_precision_smooth: 0.1622 - val_recall_smooth: 0.5780
#Epoch 41/200
#800/800 [==============================] - 350s 438ms/step - loss: 0.0764 - binary_accuracy: 0.9567 - dice_coefficient: 0.5127 - precision_smooth: 0.5074 - recall_smooth: 0.5303 - val_loss: 0.1379 - val_binary_accuracy: 0.9405 - val_dice_coefficient: 0.3503 - val_precision_smooth: 0.3408 - val_recall_smooth: 0.3670
#Epoch 42/200
#800/800 [==============================] - 346s 432ms/step - loss: 0.0724 - binary_accuracy: 0.9584 - dice_coefficient: 0.5403 - precision_smooth: 0.5295 - recall_smooth: 0.5607 - val_loss: 0.1317 - val_binary_accuracy: 0.9454 - val_dice_coefficient: 0.2950 - val_precision_smooth: 0.4048 - val_recall_smooth: 0.2355
#Epoch 43/200
#800/800 [==============================] - 346s 433ms/step - loss: 0.0709 - binary_accuracy: 0.9588 - dice_coefficient: 0.5486 - precision_smooth: 0.5396 - recall_smooth: 0.5687 - val_loss: 0.1427 - val_binary_accuracy: 0.9457 - val_dice_coefficient: 0.2918 - val_precision_smooth: 0.4672 - val_recall_smooth: 0.2135
#Epoch 44/200
#800/800 [==============================] - 331s 414ms/step - loss: 0.0698 - binary_accuracy: 0.9590 - dice_coefficient: 0.5498 - precision_smooth: 0.5436 - recall_smooth: 0.5672 - val_loss: 0.1230 - val_binary_accuracy: 0.9407 - val_dice_coefficient: 0.4311 - val_precision_smooth: 0.3564 - val_recall_smooth: 0.5531
#Epoch 45/200
#800/800 [==============================] - 337s 421ms/step - loss: 0.0683 - binary_accuracy: 0.9590 - dice_coefficient: 0.5656 - precision_smooth: 0.5607 - recall_smooth: 0.5826 - val_loss: 0.1228 - val_binary_accuracy: 0.9452 - val_dice_coefficient: 0.3925 - val_precision_smooth: 0.3878 - val_recall_smooth: 0.4050
#Epoch 46/200
#800/800 [==============================] - 337s 421ms/step - loss: 0.0656 - binary_accuracy: 0.9602 - dice_coefficient: 0.5773 - precision_smooth: 0.5664 - recall_smooth: 0.5986 - val_loss: 0.1305 - val_binary_accuracy: 0.9458 - val_dice_coefficient: 0.3265 - val_precision_smooth: 0.4254 - val_recall_smooth: 0.2681
#Epoch 47/200
#800/800 [==============================] - 331s 413ms/step - loss: 0.0612 - binary_accuracy: 0.9610 - dice_coefficient: 0.6066 - precision_smooth: 0.6004 - recall_smooth: 0.6268 - val_loss: 0.1424 - val_binary_accuracy: 0.9412 - val_dice_coefficient: 0.3462 - val_precision_smooth: 0.3654 - val_recall_smooth: 0.3335
#Epoch 48/200
#800/800 [==============================] - 331s 414ms/step - loss: 0.0622 - binary_accuracy: 0.9610 - dice_coefficient: 0.6028 - precision_smooth: 0.5976 - recall_smooth: 0.6161 - val_loss: 0.1328 - val_binary_accuracy: 0.9371 - val_dice_coefficient: 0.3493 - val_precision_smooth: 0.3264 - val_recall_smooth: 0.3807
#Epoch 49/200
#800/800 [==============================] - 327s 408ms/step - loss: 0.0614 - binary_accuracy: 0.9609 - dice_coefficient: 0.5981 - precision_smooth: 0.5927 - recall_smooth: 0.6139 - val_loss: 0.1076 - val_binary_accuracy: 0.9477 - val_dice_coefficient: 0.4423 - val_precision_smooth: 0.4303 - val_recall_smooth: 0.4604
#Epoch 50/200
#800/800 [==============================] - 337s 422ms/step - loss: 0.0570 - binary_accuracy: 0.9626 - dice_coefficient: 0.6304 - precision_smooth: 0.6196 - recall_smooth: 0.6484 - val_loss: 0.1490 - val_binary_accuracy: 0.9477 - val_dice_coefficient: 0.3370 - val_precision_smooth: 0.5393 - val_recall_smooth: 0.2487
#Epoch 51/200
#800/800 [==============================] - 347s 434ms/step - loss: 0.0556 - binary_accuracy: 0.9631 - dice_coefficient: 0.6422 - precision_smooth: 0.6381 - recall_smooth: 0.6569 - val_loss: 0.1323 - val_binary_accuracy: 0.9446 - val_dice_coefficient: 0.3920 - val_precision_smooth: 0.4263 - val_recall_smooth: 0.3690
#Epoch 52/200
#800/800 [==============================] - 341s 426ms/step - loss: 0.0519 - binary_accuracy: 0.9640 - dice_coefficient: 0.6602 - precision_smooth: 0.6483 - recall_smooth: 0.6809 - val_loss: 0.1288 - val_binary_accuracy: 0.9501 - val_dice_coefficient: 0.4212 - val_precision_smooth: 0.5538 - val_recall_smooth: 0.3429
#Epoch 53/200
#800/800 [==============================] - 337s 421ms/step - loss: 0.0527 - binary_accuracy: 0.9636 - dice_coefficient: 0.6572 - precision_smooth: 0.6496 - recall_smooth: 0.6769 - val_loss: 0.1671 - val_binary_accuracy: 0.9419 - val_dice_coefficient: 0.3292 - val_precision_smooth: 0.3998 - val_recall_smooth: 0.2853
#Epoch 54/200
#800/800 [==============================] - 388s 485ms/step - loss: 0.0528 - binary_accuracy: 0.9637 - dice_coefficient: 0.6615 - precision_smooth: 0.6528 - recall_smooth: 0.6833 - val_loss: 0.1465 - val_binary_accuracy: 0.9479 - val_dice_coefficient: 0.3615 - val_precision_smooth: 0.5382 - val_recall_smooth: 0.2765
#Epoch 55/200
#800/800 [==============================] - 414s 517ms/step - loss: 0.0488 - binary_accuracy: 0.9651 - dice_coefficient: 0.6775 - precision_smooth: 0.6752 - recall_smooth: 0.6894 - val_loss: 0.1308 - val_binary_accuracy: 0.9466 - val_dice_coefficient: 0.4489 - val_precision_smooth: 0.4880 - val_recall_smooth: 0.4195
#Epoch 56/200
#800/800 [==============================] - 341s 426ms/step - loss: 0.0502 - binary_accuracy: 0.9645 - dice_coefficient: 0.6799 - precision_smooth: 0.6721 - recall_smooth: 0.7019 - val_loss: 0.1383 - val_binary_accuracy: 0.9487 - val_dice_coefficient: 0.4071 - val_precision_smooth: 0.5321 - val_recall_smooth: 0.3344
#Epoch 57/200
#800/800 [==============================] - 329s 411ms/step - loss: 0.0465 - binary_accuracy: 0.9657 - dice_coefficient: 0.6924 - precision_smooth: 0.6852 - recall_smooth: 0.7071 - val_loss: 0.1409 - val_binary_accuracy: 0.9442 - val_dice_coefficient: 0.4298 - val_precision_smooth: 0.4733 - val_recall_smooth: 0.3977
#Epoch 58/200
#800/800 [==============================] - 329s 412ms/step - loss: 0.0457 - binary_accuracy: 0.9657 - dice_coefficient: 0.7024 - precision_smooth: 0.6909 - recall_smooth: 0.7200 - val_loss: 0.1489 - val_binary_accuracy: 0.9454 - val_dice_coefficient: 0.3757 - val_precision_smooth: 0.4781 - val_recall_smooth: 0.3130
#Epoch 59/200
#800/800 [==============================] - 331s 414ms/step - loss: 0.0450 - binary_accuracy: 0.9660 - dice_coefficient: 0.7081 - precision_smooth: 0.7084 - recall_smooth: 0.7177 - val_loss: 0.1724 - val_binary_accuracy: 0.9395 - val_dice_coefficient: 0.3080 - val_precision_smooth: 0.3829 - val_recall_smooth: 0.2614
#Epoch 60/200
#800/800 [==============================] - 358s 447ms/step - loss: 0.0437 - binary_accuracy: 0.9664 - dice_coefficient: 0.7146 - precision_smooth: 0.7045 - recall_smooth: 0.7290 - val_loss: 0.1430 - val_binary_accuracy: 0.9460 - val_dice_coefficient: 0.4206 - val_precision_smooth: 0.4885 - val_recall_smooth: 0.3733
#Epoch 61/200
#800/800 [==============================] - 353s 442ms/step - loss: 0.0441 - binary_accuracy: 0.9662 - dice_coefficient: 0.7138 - precision_smooth: 0.7081 - recall_smooth: 0.7269 - val_loss: 0.1382 - val_binary_accuracy: 0.9398 - val_dice_coefficient: 0.4537 - val_precision_smooth: 0.4160 - val_recall_smooth: 0.5061
#Epoch 62/200
#800/800 [==============================] - 341s 426ms/step - loss: 0.0431 - binary_accuracy: 0.9664 - dice_coefficient: 0.7174 - precision_smooth: 0.7085 - recall_smooth: 0.7329 - val_loss: 0.1993 - val_binary_accuracy: 0.9449 - val_dice_coefficient: 0.2218 - val_precision_smooth: 0.5714 - val_recall_smooth: 0.1407
#Epoch 63/200
#800/800 [==============================] - 339s 424ms/step - loss: 0.0412 - binary_accuracy: 0.9671 - dice_coefficient: 0.7314 - precision_smooth: 0.7289 - recall_smooth: 0.7421 - val_loss: 0.1517 - val_binary_accuracy: 0.9488 - val_dice_coefficient: 0.3895 - val_precision_smooth: 0.5926 - val_recall_smooth: 0.2927
#Epoch 64/200
#800/800 [==============================] - 356s 445ms/step - loss: 0.0405 - binary_accuracy: 0.9673 - dice_coefficient: 0.7348 - precision_smooth: 0.7271 - recall_smooth: 0.7465 - val_loss: 0.1550 - val_binary_accuracy: 0.9447 - val_dice_coefficient: 0.3603 - val_precision_smooth: 0.4470 - val_recall_smooth: 0.3047
#Epoch 65/200
#800/800 [==============================] - 428s 536ms/step - loss: 0.0404 - binary_accuracy: 0.9672 - dice_coefficient: 0.7352 - precision_smooth: 0.7305 - recall_smooth: 0.7467 - val_loss: 0.1407 - val_binary_accuracy: 0.9483 - val_dice_coefficient: 0.4596 - val_precision_smooth: 0.5477 - val_recall_smooth: 0.4013
#Epoch 66/200
#800/800 [==============================] - 390s 487ms/step - loss: 0.0397 - binary_accuracy: 0.9675 - dice_coefficient: 0.7388 - precision_smooth: 0.7323 - recall_smooth: 0.7548 - val_loss: 0.2080 - val_binary_accuracy: 0.9424 - val_dice_coefficient: 0.2745 - val_precision_smooth: 0.4939 - val_recall_smooth: 0.1935
#Epoch 67/200
#800/800 [==============================] - 349s 436ms/step - loss: 0.0403 - binary_accuracy: 0.9674 - dice_coefficient: 0.7396 - precision_smooth: 0.7402 - recall_smooth: 0.7480 - val_loss: 0.2529 - val_binary_accuracy: 0.9427 - val_dice_coefficient: 0.1425 - val_precision_smooth: 0.5270 - val_recall_smooth: 0.0834
#Epoch 68/200
#800/800 [==============================] - 351s 439ms/step - loss: 0.0371 - binary_accuracy: 0.9684 - dice_coefficient: 0.7575 - precision_smooth: 0.7506 - recall_smooth: 0.7716 - val_loss: 0.2130 - val_binary_accuracy: 0.9441 - val_dice_coefficient: 0.2280 - val_precision_smooth: 0.5472 - val_recall_smooth: 0.1452
#Epoch 69/200
#800/800 [==============================] - 349s 437ms/step - loss: 0.0366 - binary_accuracy: 0.9685 - dice_coefficient: 0.7610 - precision_smooth: 0.7557 - recall_smooth: 0.7720 - val_loss: 0.1528 - val_binary_accuracy: 0.9353 - val_dice_coefficient: 0.4516 - val_precision_smooth: 0.3972 - val_recall_smooth: 0.5315
#100/100 [==============================] - 12s 118ms/step
#Test loss, acc, dice, precision, recall): [0.13303620278835296, 0.9445749378204346, 0.4889341735839844, 0.412155796289444, 0.6012316966056823]
#------------------------------
#T:  3
#Exec time per prediction = 0.437
#aleatoric:  0.017124258 0.005948321
#epistemic:  0.004043054 0.0024030409
#Dice:  0.9445635277032852 0.02368706212526368
#------------------------------
#------------------------------
#T:  5
#Exec time per prediction = 0.647
#aleatoric:  0.01690114 0.0060521006
#epistemic:  0.004684251 0.0026255127
#Dice:  0.9441615134477616 0.023519920590633094
#------------------------------
#------------------------------
#T:  10
#Exec time per prediction = 1.292
#aleatoric:  0.0170424 0.0058184224
#epistemic:  0.0053774132 0.0024535144
#Dice:  0.9456225568056107 0.02200191293384321
#------------------------------
#------------------------------
#T:  15
#Exec time per prediction = 1.736
#aleatoric:  0.016942717 0.0058612465
#epistemic:  0.0055793053 0.0028490217
#Dice:  0.9429782426357269 0.024033580601382408
#------------------------------
#------------------------------
#T:  20
#Exec time per prediction = 2.460
#aleatoric:  0.01697247 0.00569916
#epistemic:  0.0057938667 0.0027421424
#Dice:  0.943761123418808 0.02473216667748335
#------------------------------
#------------------------------
#T:  30
#Exec time per prediction = 3.539
#aleatoric:  0.016918546 0.0058083143
#epistemic:  0.005800209 0.0028638428
#Dice:  0.9429595297574997 0.02386772403299037
#------------------------------
#Exec time per prediction = 0.661
#aleatoric:  0.016773479
#epistemic:  0.0048302324
#aleatoric mean:  0.016773479
#epistemic mean:  0.0048302324
#D:\Program Files\Spyder\WPy64-3740\python-3.7.4.amd64\lib\site-packages\seaborn\distributions.py:679: UserWarning: Passing a 2D dataset for a bivariate plot is deprecated in favor of kdeplot(x, y), and it will cause an error in future versions. Please update your code.
#  warnings.warn(warn_msg, UserWarning)
#0.19583481758832927 0.21137385569512832 0.23206647324562146 0.23691193868220145
#conditional expectation of epistemic
#0.030781705 0.051362682 0.044989653 0.023944762