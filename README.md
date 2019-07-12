# Collision

**Collsion** Proof of concept for a neural network to detect collsions of bodies. 
For training, validation and testing we us Blender generated images (size: 540 x 540 x 3) of cubes. 
It uses as LeNet like architecture and the whole can be used as a template for other CNN projects.

## Usage
```shell
usage: collision.py [-h] [--shuffle] [--ptrain PTRAIN] [--pval PVAL]
                    [--ptest PTEST] [--epochs EPOCHS] [--training] [--testing]
                    [--predict PREDICT] [--verbose]

optional arguments:
  -h, --help         show this help message and exit
  --shuffle          shuffle the data before processing
  --ptrain PTRAIN    training percentage
  --pval PVAL        validation percentage
  --ptest PTEST      test percentage
  --epochs EPOCHS    number of epochs of training
  --training         call the training function
  --testing          call the testing function
  --predict PREDICT  prints the prediction array for an image to stdout
  --verbose          more verbose output to stdout
```

One or more of the following four attributes must be given. The other attributes are optional
* --training

* --testing: determines the quality of the trained model. Output is a dictionary with the summary of correct and false detections per category (in, out, collision). The only prerequisite for this option is that the model has been trained before (e.g by using the --training option) 

* --shuffle

* --predict

optional parameters are
* --verbose

* --pval

* --ptrain

* --ptest

* --help

## Credits
Credits go to
* Blender.org offering this georguous program
* My son Leif who generated the images 
