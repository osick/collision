
import sys
import os
import argparse
import json


if __name__ == "__main__":
    # Parser setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--shuffle",     help="shuffle the data before processing",         action="store_true")
    parser.add_argument("--epochs",      help="number of epochs of training (default:10)",  type=int,   default=10)
    parser.add_argument("--model",       help="choose a model (default: Conf95)",           type=str,   default="Conf95")
    parser.add_argument("--training",    help="call the training function",                 action="store_true")
    parser.add_argument("--testing",     help="call the testing function",                  action="store_true")
    parser.add_argument("--predict",     help="compute the prediction of an image", type=str)
    args=parser.parse_args()
    # determines which routine shold be done 
    if not (args.training or args.testing or args.shuffle or args.predict): 
        parser.print_help()
    else:
        import collision
        model_path="models"
        with open(os.path.join(model_path, args.model+".model"),"r") as m: 
            mdl=json.load(m)
        cData  = collision.collisionData("data")
        cModel = collision.collisionModel(cData, mdl, "models")
        cData.dataConfiguration["percentage"] = cModel.dataConfiguration["percentage"]
        if args.shuffle:    cData.shuffle(cModel.dataConfiguration["percentage"])
        if args.training:   cModel.training(args.epochs)
        if args.testing:    cModel.test()    
        if args.predict:    cModel.predict(args.predict)
    