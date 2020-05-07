#!/usr/bin/env python3
#-*-coding:utf-8-*-

import os

#const
log_path  = "data"




# setup the logger configuration
import logging
if not os.path.isdir(log_path): os.mkdir(log_path)
logging.basicConfig(filename=os.path.join(log_path,"collision.info.log"),filemode="a+",level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger=logging.getLogger(__name__)

# some std libs 
import numpy as np
import sys
import shutil
import json
import zipfile
import re


# some AI libs
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras import optimizers
from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras import callbacks

# make tf not so noisy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class collisionData():
    
    def __init__(self, dataPath):
        self.categories         = ["collision", "in", "out"]
        self.data_path          = dataPath 
        self.img_data           = {"img_width":108, "img_height":108, "channels":3}
        self.dataConfiguration  = {"percentage":{}}

    def prepare(self, percentages, filename="full.zip"):
        logger.info("starting prepare()...")
        #unzip data
        with zipfile.ZipFile(os.path.join(self.data_path,filename),"r") as zip_ref:
            zip_ref.extractall(self.data_path)
        logger.info("file "+self.data_path+" "+filename +" unzipped to "+self.data_path)
        #generate sub dirs for data
        for type in percentages.keys():
            type_path=os.path.join(self.data_path,type)
            print("prepare: type",type_path)
            if not os.path.isdir(type_path):
                os.mkdir(type_path)
                logger.info("directory "+type_path+" generated")
            for cls in self.categories:
                cls_path=os.path.join(type_path,cls)
                if not os.path.isdir(cls_path):  
                    os.mkdir(cls_path)
                    logger.info("directory "+cls_path+" generated")
        with open(".prepared", "w") as prep: 
            prep.write("data and dir prepared")    


    def shuffle(self, percentages):
        ''' separating img data in three different sub dirs: "train", "val", "test", the size of these sample sets is given by the relevant args '''

        #if not os.path.isfile(".prepared"): self.prepare(percentages) # prepare Data only on first usage
        for category in self.categories:
            source_path=os.path.join(self.data_path,"full",category)
            files = [f for f in os.listdir(source_path)]
            np.random.shuffle(files)
            logger.info(category+ " files:" + str(len(files)))
            count=0
            for type, perc in percentages.items():
                target_path =os.path.join(self.data_path,type,category)
                shutil.rmtree(target_path)
                os.mkdir(target_path)
                upper = count+int(len(files)*perc)
                logger.info(str(int(len(files)*perc)) + " files to "+ target_path)
                for file in files[count:upper]: 
                    shutil.copy(os.path.join(source_path,file), os.path.join(target_path,file))
                count = upper


class collisionModel():

    def __init__(self,cd, modelConf):
        self.cData = cd
        self.dataConfiguration={"percentage":{"train":0.75,"val":0.15,"test":0.1}}
        self.stages = {stage:os.path.join(self.cData.data_path,stage) for stage in ["train","val","test"]}
        self.modelConfiguration = modelConf
        self.model_path = "models"
        if not os.path.isdir(self.model_path): os.mkdir(self.model_path)
        self.model_filepath = os.path.join(self.model_path,"model"+self.modelConfiguration['name']+".h5")
        self.weights_filepath = os.path.join(self.model_path,"weights"+self.modelConfiguration['name']+".h5")
        
    def initModel(self):    
        ''' LeNet-5 type CNN model'''
        # init sequential model
        self.model=Sequential()
        #loop the modelconfiguration["steps"]
        for i,step in enumerate(self.modelConfiguration["steps"]):
            print("step",i,step)
            if step["type"]=="Conv2D":
                if i==0: self.model.add(Conv2D( step["filters"], step["kernel-size"] ,input_shape=(self.cData.img_data["img_width"],self.cData.img_data["img_height"],self.cData.img_data["channels"]),activation=step["activation"]))
                else: self.model.add(Conv2D( step["filters"], step["kernel-size"] ,activation=step["activation"]))
            elif step["type"]=="MaxPooling2D":
                if "data_format" in step: self.model.add(MaxPooling2D(pool_size=step["pool-size"],data_format=step["data_format"]))
                else: self.model.add(MaxPooling2D(pool_size=step["pool-size"]))
            elif step["type"]=="Flatten":           self.model.add(Flatten())
            elif step["type"]=="AveragePooling2D":  self.model.add(AveragePooling2D())
            elif step["type"]=="Dense":             self.model.add(Dense(step["units"],activation=step["activation"]))
            elif step["type"]=="Dropout":           self.model.add(Dropout(step["rate"]))
            else: print("model step", step ,"unknown")
        # compile the model        
        self.model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])

    def training(self,epochs=10, cleanupBefore=True):
        ''' train the model'''

        # cleanup old data
        for stage in self.stages:
            pluginDir=os.path.join(self.stages[stage],"plugins") 
            if os.path.isdir(pluginDir): shutil.rmtree(pluginDir)
            for f in os.listdir(self.stages[stage]): 
                if re.search("^events", f): 
                    os.remove(os.path.join(self.stages[stage], f))
        if os.path.isdir(os.path.join(self.cData.data_path,"validation")): 
            shutil.rmtree(os.path.join(self.cData.data_path,"validation"))
            
        
        train_datagen   = ImageDataGenerator( rescale = 1./255)
        train_generator = train_datagen.flow_from_directory(self.stages["train"], target_size=(self.cData.img_data["img_width"], self.cData.img_data["img_height"]), batch_size=32, class_mode='categorical')
        val_datagen     = ImageDataGenerator( rescale = 1./127)
        val_generator   = val_datagen.flow_from_directory(self.stages["val"], target_size=(self.cData.img_data["img_width"], self.cData.img_data["img_height"]), batch_size=32, class_mode='categorical')
        self.initModel()
        tensorboard_callback = callbacks.TensorBoard(log_dir=log_path)
        self.model.fit_generator(train_generator, samples_per_epoch=2000, epochs=epochs, validation_data=val_generator, validation_steps=32,callbacks=[tensorboard_callback])
        self.model.save(self.model_filepath)
        self.model.save_weights(self.weights_filepath)

    def test(self):
        ''' test the model '''
        
        _falseCount=0
        _totalCount=1
        _total={key:{key1:0 for key1 in self.cData.categories} for key in self.cData.categories}
        model = load_model(self.model_filepath)
        model.load_weights(self.weights_filepath)
        for category in self.cData.categories:
            cat_path=os.path.join(self.stages["test"],category)
            print(cat_path)    
            _totalCount+=len(os.listdir(cat_path))
            for filename in os.listdir(cat_path):
                x = load_img(os.path.join(cat_path,filename), target_size=(self.cData.img_data["img_width"], self.cData.img_data["img_height"]))
                array = model.predict(np.expand_dims(img_to_array(x), axis=0))
                result=self.cData.categories[np.argmax(array[0])]
                _total[category][result]+=1
                if not result==category: 
                    _falseCount=_falseCount+1 
                    logger.debug("false detection:" + cat_path + filename + ":" + result)
        sys.stdout.write("Detail result\n" + json.dumps(_total,indent=2) + "\n")
        sys.stdout.write("Total detection rate: {:05.1f}%\n\n".format((_totalCount - _falseCount)/_totalCount*100))
        sys.stdout.flush()

    def predict(self, filepath):
        '''returns prediction values array for a single image'''

        # bad style, but libs are only needed here, so we load it only in this routine
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        # init
        model = load_model(self.model_filepath)
        model.load_weights(self.weights_filepath)
        x = load_img(filepath, target_size=(self.cData.img_data["img_width"], self.cData.img_data["img_height"]))
        # main step: Prediction from model
        arr=model.predict(np.expand_dims(img_to_array(x), axis=0))[0]
        logger.info(arr)
        predicted_category=self.cData.categories[np.argmax(arr)]
        # result to stdout
        result="image " + filepath + " is of type '"+predicted_category+"'"
        logger.info(result)
        sys.stdout.write("\n"+result+"\n"+"-"*len(result)+"\n")
        for i, prediction in enumerate(arr): sys.stdout.write("P('{}') = {:.4f}\n".format(self.cData.categories[i],prediction))
        sys.stdout.flush()
        # show image
        img=mpimg.imread(filepath)
        _ = plt.imshow(img)
        plt.title("predicted type: '" + predicted_category + "'")
        plt.show()
