# Collision
![collisions](banner.jpg)

**Collsion** is a simple proof of concept of a neural network detecting collsions of bodies. In this setup it can reach an accuracy of about 98% which sounds good, but in fact is not good enough remembering self driving cars ...  

For training, validation and testing it uses abstract Blender generated images (size: 540 x 540 x 3) of cubes. 

Futher it uses an LeNet-5 like architecture (for more see [convolutional neural networks](https://en.wikipedia.org/wiki/Convolutional_neural_network) on Wikipedia). 

The whole can be used as a template for other CNN projects. Feel free.

## Usage
There are 5 steps to use it correctly

1. After downloading the whole repository, unpack the file full.zip in the data directory

2. Shuffle the data by `$ collision.py --shuffle`

3. The model has to be trained `$ collision.py --training`

4. The model has to be tested `$ collision.py --testing`
   testing determines the quality of the trained model. Output is a dictionary with the summary of correct and false detections per category (in, out, collision). The only prerequisite for this option is that the model has been trained before (e.g by using the --training option). 
   The commands can also be combined, e.g `$ collision.py --shuffle --trainig --testing --predict`

5. The model can now be used to make prediction (remember the image should have resolution  540 x 540 x 3) 
  `$ collision.py --predict '/path/to/the/image' `

The whole commands are given by the following:

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

## Credits
Credits go to
* blender.org offering this georguous program
* My son who generated the images 
