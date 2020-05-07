
import sys
import argparse

modelConf95={
    "name":"Conf95",
    "steps":[
        {"type":"Conv2D",           "filters":48, "kernel-size":(3,3), "activation":"relu"}, 
        {"type":"MaxPooling2D",     "pool-size":(2,2)}, 
        {"type":"Conv2D",           "filters":16, "kernel-size":(3,3), "activation":"relu"}, 
        {"type":"MaxPooling2D",     "pool-size":(2,2), "data_format":"channels_first"}, 
        {"type":"Flatten"}, 
        {"type":"Dense",            "units":256, "activation":"relu"}, 
        {"type":"Dropout",          "rate":0.25}, 
        {"type":"Dense",            "units":32, "activation":"relu"}, 
        {"type":"Dropout",          "rate":0.25}, 
        {"type":"Dense",            "units":3, "activation":"softmax"}
    ]
}

modelLeNet5={
    "name":"LeNet5",
    "steps":[   
        {"type":"Conv2D",   "filters":6, "kernel-size":(3,3), "activation":"relu"}, 
        {"type":"AveragePooling2D"}, 
        {"type":"Conv2D",   "filters":16, "kernel-size":(3,3), "activation":"relu"},
        {"type":"AveragePooling2D"},  
        {"type":"Flatten"}, 
        {"type":"Dense",    "units":120,  "activation":"relu"},
        {"type":"Dense",    "units":84,   "activation":"relu"},
        {"type":"Dense",    "units":3,   "activation":"softmax"}
    ]
}

if __name__ == "__main__":
    # Parser setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--shuffle",     help="shuffle the data before processing", action="store_true")
    parser.add_argument("--epochs",      help="number of epochs of training",       type=int)
    parser.add_argument("--training",    help="call the training function",         action="store_true")
    parser.add_argument("--testing",     help="call the testing function",          action="store_true")
    parser.add_argument("--predict",     help="compute the prediction of an image")
    args=parser.parse_args()
    # determines which routine shold be done 
    if not (args.training or args.testing or args.shuffle or args.predict): 
        parser.print_help()
    else:
        import collision
        cData  = collision.collisionData("data")
        cModel = collision.collisionModel(cData, modelConf95)
        cData.dataConfiguration["percentage"]=cModel.dataConfiguration["percentage"]
        if args.shuffle:    cData.shuffle(cModel.dataConfiguration["percentage"])
        if args.training:   cModel.training(args.epochs if args.epochs else 10)
        if args.testing:    cModel.test()    
        if args.predict:    cModel.predict(args.predict)
