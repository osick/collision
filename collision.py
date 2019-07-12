#!/usr/bin/env python3
#-*-coding:utf-8-*-

#const
categories          = ["collision", "in", "out"]
log_path            = "./logs"
model_path          = "./models"
data_path           = "./data"
model_filepath      = model_path+'/model.h5'
weights_filepath    = model_path+'/weights.h5'
stages              = {"train":data_path+"/train",   "val":data_path+"/val",   "test":data_path+"/test"}
img_data            = {"img_width":108, "img_height":108, "channels":3}

# setup the logger configuration
import logging
logging.basicConfig(filename=log_path+"/collision.info.log",filemode="a+",level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger=logging.getLogger(__name__)

# some std libs 
import os
import time
import numpy as np
import sys
import shutil
import json

# some AI libs
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras import optimizers
from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import callbacks

# make tf not so noisy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def define_model():    
    '''
    LeNet type CNN model
    '''
    model=Sequential()
    model.add(Conv2D(48,(3,3),input_shape=(img_data["img_width"],img_data["img_height"],img_data["channels"]),activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16,(3,3),activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2),data_format="channels_first"))
    model.add(Flatten())
    model.add(Dense(units=256,activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(units=32,activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(3,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
    return model


def prepare():
    import zipfile
    logger.info("starting prepare()...")

    #unzip data
    with zipfile.ZipFile(data_path+"full.zip","r") as zip_ref:
        zip_ref.extractall(data_path)
    logger.info("file "+data_path+"full.zip unzipped to "+data_path)
    
    #generate sub dirs for data
    for type in ["train","val","test"]:
        type_path=data_path+"/"+type
        if not os.path.isdir(type_path):
            os.mkdir(type_path)
            logger.info("directory "+type_path+" generated")
        for cls in ["in","out","collision"]:
            cls_path=type_path+"/"+cls
            if not os.path.isdir(cls_path):  
                os.mkdir(cls_path)
                logger.info("directory "+cls_path+" generated")
    prep = open("./.prepared", "w")
    prep.write("data and dir prepared")
    prep.close()

    #generate log dir and model dir
    if not os.path.isdir(log_path):   
        os.mkdir(log_path)
        logger.info("directory "+log_path+" generated")

    if not os.path.isdir(model_path): 
        os.mkdir(model_path)
        logger.info("directory "+model_path+" generated")
    logger.info("prepare() done.")


def shuffle_data(p_tr=0.75, p_va=0.15, p_te=0.1):
    '''
    separating img data in three different sub dirs: 
    - "train"
    - "val"
    - "test"
    the size of these sample sets is given by the relevant args
    remark: No dependency on keras
    '''

    if not os.path.isfile("./.prepared"):
        prepare()
    perc_vals = {"train":p_tr, "val":p_va, "test":p_te}
    for category in categories:
        source_path="."+os.sep+"data"+os.sep+"full"+os.sep+category+os.sep
        files = [f for f in os.listdir(source_path)]
        np.random.shuffle(files)
        logger.info(category+ " files:" + str(len(files)))
        count=0
        for type, perc in perc_vals.items():
            target_path = "."+os.sep+"data"+os.sep+type+os.sep+category+os.sep
            shutil.rmtree(target_path)
            os.mkdir(target_path)
            upper = count+int(len(files)*perc)
            logger.info(str(int(len(files)*perc)) + " files to "+ target_path)
            for file in files[count:upper]:
                shutil.copy(source_path+file, target_path+file)
            count = upper


def training(epochs=10):
    '''
    train the model
    '''
    train_datagen   = ImageDataGenerator( rescale = 1./255)
    train_generator = train_datagen.flow_from_directory(stages["train"], target_size=(img_data["img_width"], img_data["img_height"]), batch_size=32, class_mode='categorical')
    val_datagen     = ImageDataGenerator( rescale = 1./127)
    val_generator   = val_datagen.flow_from_directory(stages["val"], target_size=(img_data["img_width"], img_data["img_height"]), batch_size=32, class_mode='categorical')
    model = define_model()
    tensorboard_callback = callbacks.TensorBoard(log_dir=log_path)
    model.fit_generator(train_generator, samples_per_epoch=2000, epochs=epochs, validation_data=val_generator, validation_steps=32,callbacks=[tensorboard_callback])
    model.save(model_filepath)
    model.save_weights(weights_filepath)


def testing(verbose=False):
    '''
    test the model
    '''
    _f=0
    _t=0
    _total={ "in":{"in":0,"collision":0,"out":0},  "collision":{"in":0,"collision":0,"out":0},  "out":{"in":0,"collision":0,"out":0} }
    model = load_model(model_filepath)
    model.load_weights(weights_filepath)
    for category in categories:
        cat_path=stages["test"]+"/"+category+"/"    
        _t+=len(os.listdir(cat_path))
        for filename in os.listdir(cat_path):
            x = load_img(cat_path+filename, target_size=(img_data["img_width"], img_data["img_height"]))
            array = model.predict(np.expand_dims(img_to_array(x), axis=0))
            result=categories[np.argmax(array[0])]
            _total[category][result]+=1
            if result!=category:
                _f=_f+1
                logger.debug("false detection:" + cat_path + filename + ":" + result)
                if verbose:
                    sys.stdout.write("false detection:" + cat_path + filename + ":" + result+"\n")
                    sys.stdout.flush()
    sys.stdout.write("Detail result\n" + json.dumps(_total,indent=2) + "\n")
    sys.stdout.write("Total detection rate: {:05.1f}%\n\n".format((_t-_f)/_t*100))
    sys.stdout.flush()


def predict(filepath):
    '''
    returns prediction values array for a single image
    '''
    model = load_model(model_filepath)
    model.load_weights(weights_filepath)
    x = load_img(filepath, target_size=(img_data["img_width"], img_data["img_height"]))
    arr=model.predict(np.expand_dims(img_to_array(x), axis=0))
    logger.info(arr)
    sys.stdout.write(arr)
    sys.stdout.flush()

def main():
    '''
    main routine with for different use cases
    - shuffle data
    - train from data
    - test from data
    - predict from a single data set
    '''
    # Parser setup
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--shuffle",     help="shuffle the data before processing", action="store_true")
    parser.add_argument("--ptrain",      help="training percentage", type=float)
    parser.add_argument("--pval",        help="validation percentage", type=float)
    parser.add_argument("--ptest",       help="test percentage", type=float)
    parser.add_argument("--epochs",      help="number of epochs of training", type=int)
    parser.add_argument("--training",    help="call the training function", action="store_true")
    parser.add_argument("--testing",     help="call the testing function", action="store_true")
    parser.add_argument("--predict",     help="prints the prediction array for an image to stdout")
    parser.add_argument("--verbose",     help="more verbose output to stdout", action="store_true")
    args=parser.parse_args()

    if not (args.training or args.testing or args.shuffle or args.predict):
        sys.stderr.write("\nERROR:\n[--predict], [--shuffle], [--training] or [--testing] must be set.\nExiting...\n")
        sys.stderr.flush()
        parser.print_help()
        sys.exit(1)
    if args.predict:  
        predict(args.predict)
        sys.exit(0)
    if args.shuffle:  
        p_tr=0.75
        p_va=0.15
        p_te=0.1
        if args.ptrain:  p_tr=args.ptrain 
        if args.pval:    p_va=args.pval
        if args.ptest:   p_te=args.ptest 
        shuffle_data(p_tr, p_va, p_te)
    if args.training: 
            if args.epochs: training(args.epochs)
            else: training()
    if args.testing:  
            if args.verbose: testing(args.verbose)
            else: testing()

if __name__ == "__main__":
    main()