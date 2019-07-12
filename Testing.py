import time
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model

model_path, model_weights_path = './models/model.h5'  ,  './models/weights.h5'
test_path = './data/test/'
img_width, img_height = 180, 180
img_categories=["in","in-out","out"]

def main():
    _false=0
    _total=0
    model = load_model(model_path)
    model.load_weights(model_weights_path)
    print("\n...predicting...")
    for category in img_categories:
        cat_path=test_path+category+"/"    
        for _, ret in enumerate(os.walk(cat_path)):
            for _, filename in enumerate(ret[2]):
                x = load_img(cat_path+filename, target_size=(img_width,img_height))
                array = model.predict(np.expand_dims(img_to_array(x), axis=0))
                result=img_categories[np.argmax(array[0])]
                _total+=1
                if result!=category:
                    _false+=1
                    print("false:" + cat_path + filename + ":" + result)
    print("\ntotal: "+str(_total)+"\nfalse detections: " + str(_false) + "\ndetection_rate:" + str((_total-_false)/(_total)*100) +"%")  

if __name__=="__main__":
    start = time.time()
    main()
    end = time.time()
    print("Execution Time:",str(int(end-start)),"seconds")