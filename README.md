# image-classifier-aipnd
Image Classifier Project for Data Science Nanodegree (Udacity)

## Example Commands
```
python train.py --gpu True --learning_rate 0.002 --dropout 0.6 --epochs 3 --arch 'vgg19' --hidden_units 500
```
```
python predict.py '/home/workspace/paind-project/flowers/test/16/image_06657.jpg' --top_k 6 --gpu True
```

## Training
To train a model, run `train.py` with the desired model architecture (densenet or vgg) and the path to the image folder:

```
python train.py --arch 'densenet' 
```
The command above will use default values for all other values. See below for how to customize these values.

### Usage
```
usage: train.py [-h] [--data_dir DATA_DIR] [--save_dir SAVE_DIR] [--arch ARCH]
                [--learning_rate LEARNING_RATE] [--hidden_units HIDDEN_UNITS]
                [--epochs EPOCHS] [--gpu GPU]

Provide image_dir, save_dir, architecture, hyperparameters such as
learningrate, num of hidden_units, epochs and whether to use gpu or not

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   path to image folder
  --save_dir SAVE_DIR   folder where model checkpoints gets saved to
  --arch ARCH           choose between vgg, densenet and alexnet
  --learning_rate LEARNING_RATE
                        learning_rate for model
  --hidden_units HIDDEN_UNITS
                        hidden_units for model
  --epochs EPOCHS       epochs for model
  --gpu GPU             whether gpu should be used for or not
```

## Prediction
To make a prediction, run `predict.py` with the desired checkpoint and path to the image you want to try and predict:

```
python predict.py 
```
The command above will use default values for all other values. See below for how to customize these values.

### Usage
```
usage: predict.py [-h] [--checkpoint CHECKPOINT] [--top_k TOP_K]
                  [--category_names CATEGORY_NAMES] [--gpu GPU]
                  input_img

predict.py

positional arguments:
  input_img

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint CHECKPOINT
  --top_k TOP_K
  --category_names CATEGORY_NAMES
  --gpu GPU
```
