# Collision

**Collsion** Proof of concept for a neural network to detect collsions of bodies. 
For training, validation and testing we us Blender generated images (size: 540 x 540 x 3) of cubes. 
It uses as LeNet like architecture and the whole can be used as a template for other CNN projects.

## Usage
```shell
usage: collision.py [-h] [--shuffle] [--ptrain PTRAIN] [--pval PVAL]
                    [--ptest PTEST] [--epochs EPOCHS] [--training] [--testing]

optional arguments:
  -h, --help       show this help message and exit
  --shuffle        shuffle the data before processing
  --ptrain PTRAIN  training percentage
  --pval PVAL      validation percentage
  --ptest PTEST    test percentage
  --epochs EPOCHS  number of epochs
  --training       call the training function
  --testing        call the testing function
```

## Credits
  todo
