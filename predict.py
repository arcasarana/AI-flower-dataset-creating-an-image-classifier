import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms
import torchvision.models as models
import os
import json
from collections import OrderedDict
import torch.optim
from torch.optim import lr_scheduler
import time
import copy
import seaborn as sns
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    
    def get_input_args():
        # Create Parse using ArgumentParser
        parser = argparse.ArgumentParser(description="Arguments for Flower Classifier")
        # Command line arguments
        parser.add_argument('--device_opt', type=str, 
                            help="Use GPU if available?  Type 'y' for 'yes' or 'n' for 'no'.",
                            default='y')
        parser.add_argument('--img', type=str, 
                            help="Path to image file to classify.",
                            default='ImageClassifier/flowers/test/16/image_06657.jpg')
        parser.add_argument('--labels', type=str, 
                            help="Path to labels to map.",
                            default="ImageClassifier/cat_to_name.json")
        parser.add_argument('--topk', type=int, 
                            default=5, 
                            help="Show how many top predictions.")
        parser.add_argument('--arch', type=str, 
                        default='densenet161', 
                        help="Select a pre-trained network (vgg16 or densenet161")
        in_args = parser.parse_args()
        
        return in_args
    
    
    def set_device(device_option):
        
        if device_option == 'n':
            device = torch.device('cpu')
            print('Using {}.'.format(device.type))
        else:
            device = torch.device("cuda" if torch.cuda.is_available()
                         else "cpu")
            if device.type == 'cuda':
                print('Using {}.'.format(device.type))
            else:
                print('GPU not available.  Using CPU.')
                
        return device
    
    def select_model(model_name):
        
        if model_name == 'vgg16':
            model = models.vgg16(pretrained=True)
            print("Using vgg16.")
        elif model_name == 'densenet161':
            model = models.densenet161(pretrained=True)
            print("Using densenet161.")

        return model
    
    def load_checkpoint(model, filepath='checkpoint.pth'):

        # Load saved file
        checkpoint = torch.load(filepath)
        # Freeze parameters
        for param in model.parameters():
            param.requires_grad = False

        model.classifier = checkpoint['classifier']
        optimizer = checkpoint['optimizer']
        epochs = checkpoint['epochs']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']

        return model
    
    def process_image(image_path):
        '''Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Numpy array
        '''
        image = Image.open(image_path)
        orig_w, orig_h = image.size

        if orig_w < orig_h:
            new_w = 256
            new_h = int((new_w * orig_h) / orig_w)
        else:
            new_h = 256
            new_w = int((new_h * orig_w) / orig_h)

        image = image.resize((new_w, new_h))

        crop_w = 224
        crop_h = 224

        left = (new_w - crop_w)/2
        top = (new_h - crop_h)/2
        right = (new_w + crop_w)/2
        bottom = (new_h + crop_h)/2

        image = image.crop((left, top, right, bottom))

       # Normalize
        image = np.array(image)/255
        mean = np.array([0.485, 0.456, 0.406]) #provided mean
        std = np.array([0.229, 0.224, 0.225]) #provided std
        image = (image - mean)/std

        # Move color channels to first dimension
        np_image = image.transpose((2, 0, 1))

        return np_image

    def label_mapping(labels):
        
        with open(labels, 'r') as f:
            cat_to_name = json.load(f)
            
        return cat_to_name
    
    def predict(image_path, model, k_num, labels, device):
        model.eval()

        # Process image
        img = process_image(image_path)

        # Numpy to Tensor
        image_tensor = torch.from_numpy(img)
        image_tensor = image_tensor.float()
        image_tensor = image_tensor.unsqueeze(0)

        # Top probabilities
        model.to(device)
        probs = torch.exp(model.forward(image_tensor.to(device)))
        top_probs, top_labs = probs.topk(k_num)
        top_probs = top_probs.detach().cpu().numpy().tolist()[0] 
        top_labs = top_labs.detach().cpu().numpy().tolist()[0]

        # Convert indices to classes
        idx_to_class = {val: key for key, val in
                        model.class_to_idx.items()}
        top_labels = [idx_to_class[lab] for lab in top_labs]
        top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]
    
        print("Here are the top {} probabilities.".format(k_num))
        
        i=0
        while i < k_num:
            print("Flower:  {},  Probability:  {}"
                  .format(top_flowers[i],top_probs[i]))
            i +=1
        return None        

    in_args = get_input_args()
    device = set_device(in_args.device_opt)
    model = select_model(in_args.arch)
    saved_checkpoint = load_checkpoint(model, filepath='checkpoint.pth')
    k_num = in_args.topk
    labels = in_args.labels
    image_path = in_args.img
    cat_to_name = label_mapping(labels)
    predict(image_path, model, k_num, labels, device)

# Call to main function to run the program
if __name__ == "__main__":
    main()
    
# Sources:
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# https://medium.com/datadriveninvestor/creating-a-pytorch-image-classifier-da9db139ba80
# https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad