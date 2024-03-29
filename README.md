# Collision
![collisions](docs/banner.jpg)

**Collision** is a simple **proof of concept** of a neural network detecting collsions of bodies (see the images above). 
In this setup it can reach an accuracy of about 98% which sounds good, but in fact is not good enough remembering self driving cars ...  

For training, validation and testing it uses abstract Blender generated images (size: 540 x 540 x 3) of cubes. 

Futher it can use an LeNet-5 like architecture (for more see [convolutional neural networks](https://en.wikipedia.org/wiki/Convolutional_neural_network) on Wikipedia). 

The whole can be used as a template for other CNN projects. Feel free.

The repo comes with a [large data file](data/full.zip) with thousand of generated images of two cubes in three different categories:
1. collision: The two cubes toch each other but don't contain themself
2. in: One cube contains the other
3. out: The cubes are separated

See the three different parts of the image above...

## Usage

Thsi repo cames with two Python files:
* collision.py: contains two class, one for handlich the data, one for the CNN
* main.py:      This contains a reference implementation of the collision classes

Usage main.py is given by the following:

```shell
usage: main.py [-h] [--shuffle] [--epochs EPOCHS] [--model MODEL] [--training]
               [--testing] [--predict PREDICT]

optional arguments:
  -h, --help         show this help message and exit
  --shuffle          shuffle the data before processing
  --epochs EPOCHS    number of epochs of training (default:10)
  --model MODEL      choose a model (default: Conf95)
  --training         call the training function
  --testing          call the testing function
  --predict PREDICT  compute the prediction of an image
```
There are 5 steps to use it:

0. Downloading the whole repository, 
1. Download full.zip and put it in the top directory of the project

1. Shuffle the data by `$ python main.py --shuffle`

2. The model has to be trained `$ python main.py --training`. You can define an own model using the `--model <modelname>` attribut. Default value ist Conf95. All models are storde in models dir with extension `.model` See [Conf95.model](models/Conf95.model). This model definition file is a very simple parametrizsation file for a convolutional neural network (CNN). This is a poor man attempt for a cofiguration of modelling...

3. The model has to be tested `$ python main.py --testing`
   testing determines the quality of the trained model. Output is a dictionary with the summary of correct and false detections per category (in, out, collision). The only prerequisite for this option is that the model has been trained before (e.g by using the --training option). 
   The commands can also be combined, e.g `$ python main.py --shuffle --trainig --testing --predict`

4. The model can now be used to make prediction (remember the image should have resolution  540 x 540 x 3) 
  `$ python main.py --predict '/path/to/the/image' `


## Licence
[GNU General Public License v3.0](COPYING)

## Credits
* Credits go to my son, who helped me a lot generating the image dataset **full.zip** using [Blender](http://blender.org)
