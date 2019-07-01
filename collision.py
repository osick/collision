import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--shuffle",     help="shuffle the data before processing", action="store_true")
parser.add_argument("--ptrain",      help="training percentage", type=float)
parser.add_argument("--pval",        help="validation percentage", type=float)
parser.add_argument("--ptest",       help="test percentage", type=float)
parser.add_argument("--epochs",      help="number of epochs", type=int)
parser.add_argument("--training",      help="number of epochs", action="store_true")
parser.add_argument("--testing",      help="number of epochs", action="store_true")
args=parser.parse_args()

import logging
logging.basicConfig("./logs/collision.log","a+",level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger=logging.getLogger(__name__)

import time
import os
import numpy as np
import sys
import shutil

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras import optimizers
from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import callbacks

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
categories  = ["in","in-out","out"]
stages      = {"train":"./data/train",   "val":"./data/val",   "test":"./data/test"}
model_path, model_weights_path      = './models/model.h5'  ,  './models/weights.h5'
img_data = {"img_width":180, "img_height":180, "channels":3}

def define_model():    
    '''
    LeNet type CNN model
    '''
    model = Sequential()
    model.add(Conv2D(6, (3, 3), input_shape=(img_data["img_width"],img_data["img_height"],img_data["channels"]), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
    model.add(Flatten())
    model.add(Dense(units=128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=15, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model

def shuffle_data():
    '''
    separating img data in three different sub dirs: 
    - "train"
    - "val"
    - "test"
    the size of these sample sets is given by the relevant args
    '''
    p_tr, p_va, p_te = 0.75, 0.15, 0.1
    if args.train:  p_te=args.ptrain 
    if args.val:    p_te=args.pval
    if args.test:   p_te=args.ptest 
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
    
def training():
    epochs = 8
    if args.shuffle: shuffle_data()
    if args.epochs: epochs=args.epochs
    train_datagen   = ImageDataGenerator( rescale = 1./255)
    train_generator = train_datagen.flow_from_directory(stages["train"], target_size=(img_data["img_width"], img_data["img_height"]), batch_size=32, class_mode='categorical')
    val_datagen     = ImageDataGenerator( rescale = 1./127)
    val_generator   = val_datagen.flow_from_directory(stages["val"], target_size=(img_data["img_width"], img_data["img_height"]), batch_size=32, class_mode='categorical')
    model = define_model()
    tensorboard_callback = callbacks.TensorBoard(log_dir="./logs")
    model.fit_generator(train_generator, samples_per_epoch=2000, epochs=epochs, validation_data=val_generator, validation_steps=32,callbacks=[tensorboard_callback])
    model.save(model_path)
    model.save_weights(model_weights_path)
    
def testing():
    _false=0
    _total=0
    model = load_model(model_path)
    model.load_weights(model_weights_path)
    for category in categories:
        cat_path=stages["test"]+"/"+category+"/"    
        for _, ret in enumerate(os.walk(cat_path)):
            for _, filename in enumerate(ret[2]):
                x = load_img(cat_path+filename, target_size=(img_data["img_width"], img_data["img_height"]))
                array = model.predict(np.expand_dims(img_to_array(x), axis=0))
                result=categories[np.argmax(array[0])]
                _total+=1
                if result!=category:
                    _false+=1
                    logger.debug("false detection:" + cat_path + filename + ":" + result)
    logger.info("total: "+str(_total)+"\nfalse detections: " + str(_false) + "\ndetection_rate:" + str((_total-_false)/(_total)*100) +"%")


if __name__ == "__main__":
    if not (args.training or args.testing):
        sys.stderr.write("argument [--training] or [--testing] must be given. exiting...\n")
        sys.stderr.flush()
        sys.exit(1)

    start = time.time()
    if args.training:
        sys.stdout.write("\n...training...\n")
        sys.stdout.flush()
        training()
    if args.testing:
        sys.stdout.write("\n...testing...\n") 
        sys.stdout.flush()
        testing()
    end = time.time()
    sys.stdout.write("Execution Time:",(end-start),"seconds")
    sys.stdout.flush()
