import utility
import argparse
import json
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import copy
from collections import OrderedDict


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def process_image(image_path):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch       
    model, returns an Numpy array
    '''
    # Open the image
    from PIL import Image
    img = Image.open(image_path)
    # Resize
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
    # Crop 
    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,   
                      top_margin))
    # Normalize
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406]) #provided mean
    std = np.array([0.229, 0.224, 0.225]) #provided std
    img = (img - mean)/std
    # Move color channels to first dimension as expected by PyTorch
    img = img.transpose((2, 0, 1))
    return img
    
    
    
def predict(image_path, model, num_classes,topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    ''' 
    # TODO: Implement the code to predict the class from an image file
    # Process image
    img = process_image(image_path) 
    #model to cpu
    model = model.to('cpu')
    # Numpy -> Tensor
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor) 
    # Add batch of size 1 to image
    model_input = image_tensor.unsqueeze(0)
    # Probs
    probs = torch.exp(model.forward(model_input))  
    # Top probs
    top_probs, top_labs = probs.topk(topk)
    top_probs = top_probs.detach().numpy().tolist()[0] 
    top_labs = top_labs.detach().numpy().tolist()[0]  
    # Convert indices to classes
    idx_to_class = {val: key for key, val in num_classes.items()}  
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]
    return top_probs, top_labels, top_flowers    

ap = argparse.ArgumentParser(description='predict.py')
ap.add_argument('input_img', default='/home/workspace/paind-project/flowers/test/102/image_08015.jpg',action="store", type = str)
ap.add_argument('--checkpoint', default='/home/workspace/paind-project/checkpoint.pth',action="store",type = str)
ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
ap.add_argument('--gpu', dest="gpu", action="store", default=True)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
    
pa = ap.parse_args()
path_image = pa.input_img
number_of_outputs = pa.top_k
power = pa.gpu
input_img = pa.input_img
path = pa.checkpoint

trainloader, validloader, testloader, train_dataset, valid_dataset,test_dataset,  = utility.create_loaders('/home/workspace/paind-project/flowers/')
model,arch,num_classes = utility.load_checkpoint(path)

probs, classes, top_flowers = predict(input_img, model, num_classes,number_of_outputs)
print('The output of model prediction is:')
print(probs)
print(classes)