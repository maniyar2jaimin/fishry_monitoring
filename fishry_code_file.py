"""
#---------------------------------------#
Author : Jaimin Maniyar
Environment : python3.5
Date_Created: 02 Apr 2017
Date_Modify: 05 Apr 2017
Note: All Libraries or packages used are of latest version
#---------------------------------------#
"""

# Libraries for image processing task

import cv2
import multiprocessing
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from skimage.filters import rank
from skimage.morphology import disk
from skimage import exposure
from skimage.filters import gaussian
from skimage.segmentation import active_contour

# Libraries for processing of data & other data related operation

import numpy as np
import time
import pandas as pd
import datetime
import os
import glob
import matplotlib.pyplot as plt
from operator import itemgetter
from itertools import groupby

# Libraries for Deep Learning operations

from keras.layers.convolutional import MaxPooling2D,AveragePooling2D,Conv2D,ZeroPadding2D
from keras.layers import Flatten, Dense, Activation, Dropout
from keras.optimizers import adam,SGD
from keras.activations import relu,sigmoid,softmax
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit,GroupKFold
from sklearn.neural_network._base import log_loss

"""---------------------------------------------------------------------------------"""

# Global Variable section
image_shape = []
FISH_CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

def load_images_train():

    """
        # Method for Loading Train Images to python list
        # No Parameters
        # It will return 4 things
            * x_Train -> Train Data
            * y_Train -> Train Label
            * x_train_id -> Train file ID
            * x_shape -> shape of image
        # Created on: 02 Apr 2017
        # Modified on: 05 Apr 2017
        # Author: Jaimin
        # Modifier: Jaimin
    """

    global pool
    x_train = []
    x_train_id = []
    y_train = []
    x_shape = []
    start_time = time.time()

    print("Reading train images")
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    #folders = ['new']
    for fld in folders:
        index = folders.index(fld)
        print('Loading folder {} (Index: {})'.format(fld, index))
        path = os.path.join('./train1', fld, '*.jpg')
        files = glob.glob(path)
        pool = multiprocessing.Pool(processes=8)
        for fl in files:
            flbase = os.path.basename(fl)
            img = cv2.imread(fl,cv2.IMREAD_COLOR)
            result_list = pool.map(process_image, [fl])
            x_train.append(result_list[0])
            x_train_id.append(flbase)
            y_train.append(index)
            #x_shape.append(shape)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    pool.close()
    return x_train, y_train, x_train_id


def process_image(fl):

    """
        # Helper Method for Loading Train Images to python list using multiprocessing
        # cv2 image object as parameter
        # It will return image
        # Created on: 07 Apr 2017
        # Modified on: Nil
        # Author: Jaimin
        # Modifier: None
    """
    #print("processing of images")
    print(fl)
    img = cv2.imread(fl, cv2.IMREAD_COLOR)
    resized_img = cv2.resize(img, (146, 243), interpolation=cv2.INTER_CUBIC)
    return resized_img


def pre_processing_image(img):

    """
        # Helper Method for preprocessing of Train Images. This will be use in keras imagedatagenarator
        # It will return image
        # Created on: 07 Apr 2017
        # Modified on: Nil
        # Author: Jaimin
        # Modifier: None
    """

    #print(img.shape)
    # apply gamma correction and show the images
    #adjusted = adjust_gamma(img, gamma=0.65)

    adjusted = exposure.adjust_gamma(img, gamma=1.65)
    #print(adjusted.shape)

    # log transform of image

    logarithmic_corrected = exposure.adjust_log(adjusted, 1)
    #print(logarithmic_corrected.shape)

    # denoising
    #dst2 = cv2.fastNlMeansDenoisingColored(logarithmic_corrected, None, 10, 10, 7, 21)
    #print(dst2.shape)
    dst2 = logarithmic_corrected
    return dst2


def load_images_test():

    """
        # Method for Loading Test Images to python list
        # No Parameters
        # It will return 4 things
            * x_test -> Test Data
            * x_test_id -> Test file ID
        # Created on: 02 Apr 2017
        # Modified on: 06 Apr 2017
        # Author: Jaimin
        # Modifier: Jaimin
    """

    path = os.path.join('./test','*.jpg')
    files = glob.glob(path)

    x_test = []
    x_test_id = []
    x_test_shape = []
    pool = multiprocessing.Pool(processes=8)
    for fl in files:
        print(fl)
        flbase = os.path.basename(fl)
        img = cv2.imread(fl, cv2.IMREAD_COLOR)
        img = cv2.imread(fl, cv2.IMREAD_COLOR)
        result_list = pool.map(process_image, [fl])
        x_test.append(result_list[0])
        x_test_id.append(flbase)
        #cv2.imshow("dst", dst2)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    pool.close()
    return x_test, x_test_id


def adjust_gamma(image, gamma=1.0):

    """
        -> adjust_gamma(img)
        # Method for gamma correction of images
        # Parameter: Image object
        # It will return gamma corrected image with gamma = 0.7
        # Created on: 02 Apr 2017
        # Modified on: 05 Apr 2017
        # Author: Jaimin
        # Modifier: Jaimin
    """

    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    print("inverse of gamma")
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def image_resize(image):

    """
        -> image_resize(img)
        # Method for resizing of image
        # Parameter: Image object
        # It will return resized image as object
        # Created on: 02 Apr 2017
        # Modified on: 05 Apr 2017
        # Author: Jaimin
        # Modifier: Jaimin
    """
    print("image-resizing2")

    i=0
    height,width = image.shape[:2]
    shape = [height,width]
    if len(image_shape) == 0:
        #print("Intial")
        image_shape.append(shape)
        resized = cv2.resize(image,(int(width*0.2),int(height*0.2)),interpolation=cv2.INTER_CUBIC)
    else:
        for old_shape in image_shape:
            #print("second")
            if old_shape == shape:
                i=0
                break
            else:
                i+=1
        if(i > 0):
            #print("third")
            image_shape.append(shape)
        resized = cv2.resize(image, (int(width * 0.2), int(height * 0.2)), interpolation=cv2.INTER_CUBIC)
    return resized,shape


def create_model(input_shape=None):

    """
        # Method for CNN Model Creation
        # Created on: 06 Apr 2017
        # Modified on: 07 Apr 2017
        # Author: Jaimin
        # Modifier: Jaimin
    """

    model = Sequential()
    #n,height,width,chennel = input_shape
    height = 146
    width = 243
    chennel = 3

    model.add(Conv2D(filters=4, input_shape=(width, height, chennel), kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=8, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=4,kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=8, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=4, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=8, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=16, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=8, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=8, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=16, kernel_size=(5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(8))
    model.add(Activation('softmax'))

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.87, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])
    return model


def run_cross_validation_create_models(nfolds=10):

    """
        # Method for Applying crossvalidation on train data
        # Parameter: number of folds(nfolds)
        # Created on: 06 Apr 2017
        # Modified on: Nil
        # Author: Jaimin
        # Modifier: None
    """
    print("nfold value=",nfolds)
    x_train, y_train, x_train_id = load_images_train()
    print(len(x_train))

    # input image dimensions
    batch_size = 16
    nb_epoch = 32
    random_state = 159

    seed = 7
    np.random.seed(seed)
    kfold = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=random_state)
    #kfold = StratifiedShuffleSplit(n_splits=nfolds,test_size=0.1,train_size=0.7,random_state=random_state)
    cvscores = []
    models = []

    image_array = np.asarray(x_train, dtype=np.float32)
    # print(image_array.shape)

    datagen_test = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        preprocessing_function=pre_processing_image
    )

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        preprocessing_function=pre_processing_image,
        vertical_flip=True,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    start_time = time.time()
    print("Datagen.fit started")
    datagen.fit(image_array, augment=True, rounds=3)
    print('Fit Completed: {} seconds'.format(round(time.time() - start_time, 2)))

    img_label = np_utils.to_categorical(y_train, 8)

    yfull_train = dict()
    num_fold = 0
    sum_score = 0

    for train_index, test_index in kfold.split(x_train, y_train):
        # create model
        model = create_model()
        train_x = image_array[train_index]
        train_y = img_label[train_index]
        validate_x = image_array[test_index]
        validate_y = img_label[test_index]

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(train_x), len(train_y))
        print('Split valid: ', len(validate_x), len(validate_y))

        callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=1, mode='auto')]

        model.fit_generator(generator=datagen.flow(train_x, train_y, batch_size=batch_size, shuffle=True),
                            steps_per_epoch=len(image_array)/32, epochs=nb_epoch, verbose=1,
                            callbacks=callbacks,validation_data=(validate_x,validate_y),
                            validation_steps=len(image_array)/32)

        predictions_valid = model.predict(validate_x.astype('float32'), batch_size=batch_size, verbose=1)


        score = log_loss(validate_y, predictions_valid)
        print('Score log_loss: ', score)


        sum_score += score * len(test_index)

        # Store valid predictions
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = predictions_valid[i]

        models.append(model)

    score = sum_score / len(x_train)
    print("Log_loss train independent avg: ", score)

    info_string = 'loss_' + str(score) + '_folds_' + str(10) + '_ep_' + str(28)
    return info_string,models

def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()


def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    result1.loc['image':,] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)


def run_cross_validation_process_test(info_string, models):

    """
        # Method for Applying crossvalidation on test data
        # Parameter: info_string & Keras Model Object
        # Created on: 06 Apr 2017
        # Modified on: Nil
        # Author: Jaimin
        # Modifier: None
    """

    batch_size = 64
    num_fold = 0
    yfull_test = []
    x_test_id = []
    nfolds = len(models)

    datagen_test = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        preprocessing_function=pre_processing_image
    )

    # print(image_array.shape)
    x_test, x_test_id = load_images_test()
    print(len(x_test))
    image_test_array = np.asarray(x_test, dtype=np.float32)
    start_time = time.time()
    print("Datagen.fit started")
    datagen_test.fit(image_test_array, augment=False)
    print('Fit Completed: {} seconds'.format(round(time.time() - start_time, 2)))

    for i in range(nfolds):
        model = models[i]
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))

        #test_prediction = model.predict_generator(generator=datagen_test.fit(image_test_array, seed=79),
        #                                             steps=len(image_test_array)/32, max_q_size=20, workers=8, verbose=1)

        test_prediction = model.predict(image_test_array, batch_size=batch_size, verbose=1)

        yfull_test.append(test_prediction)

    test_res = merge_several_folds_mean(yfull_test, nfolds)
    info_string = 'loss_' + info_string \
                  + '_folds_' + str(nfolds)
    create_submission(test_res, x_test_id, info_string)
    d=pd.DataFrame(test_res,columns=FISH_CLASSES)


if __name__ == '__main__':
   
    #method 3
    #print('Keras version: {}'.format(keras))
    num_folds = 7
    info_string, models = run_cross_validation_create_models(num_folds)
    run_cross_validation_process_test(info_string, models)
