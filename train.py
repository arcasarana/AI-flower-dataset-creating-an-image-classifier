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

def main():
    
    def get_input_args():
        # Create Parse using ArgumentParser
        parser = argparse.ArgumentParser(description="Arguments for Flower Classifier")
        # Command line arguments
        parser.add_argument('--device_opt', type=str, 
                    help="Use GPU if available?  Type 'y' for 'yes' or 'n' for 'no'.",
                    default='y')
        parser.add_argument('--img', type=str, 
                            default='flowers/test/102/image_08042.jpg', 
                            help="Path to image file to classify.")
        parser.add_argument('--topK', type=int, 
                            default=5, 
                            help="Show how many top predictions.")
        parser.add_argument('--arch', type=str, 
                        default='densenet161', 
                        help="Select a pre-trained network (vgg16 or densenet161")
        parser.add_argument('--learning_rate', type=float, 
                            default=0.001, 
                            help="Set learning rate as a float.")
        parser.add_argument('--epochs', type=int, 
                            default=5, 
                            help="Set the number of epochs")
        parser.add_argument('--hidden_units', type=int, 
                            default=600, 
                            help="Set number of hidden units.")
        in_args = parser.parse_args()
        
        return in_args
   
    def select_model(model_name):
        if model_name == 'vgg16':
            model = models.vgg16(pretrained=True)
            print("Using vgg16.")
        elif model_name == 'densenet161':
            model = models.densenet161(pretrained=True)
            print("Using densenet161.")

        return model

    def build_classifier(model, hidden_units=600):
        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(2208, hidden_units)),
                              ('relu1', nn.ReLU()),
                              ('Dropout1',nn.Dropout(p=0.15)),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
        
        return classifier

    def train_classifier(model, criterion, optimizer, scheduler, epochs=5, device='cuda'):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch+1, epochs))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'valid']:
                if phase == 'train':
                    scheduler.step()
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        #load best model weights
        model.load_state_dict(best_model_wts)
        model.to(device)
        
        return model

    def save_checkpoint():
        model.class_to_idx = image_datasets['train'].class_to_idx
        checkpoint = {'input_size': 2208,
                      'output_size': 102,
                      'classifier': classifier,
                      'optimizer': optimizer,
                      'epochs': epochs,
                      'state_dict': model.state_dict(),
                      'class_to_idx': model.class_to_idx,}

        torch.save(checkpoint, 'checkpoint.pth')
        
        return checkpoint
    
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

    # Load data
    data_dir = 'ImageClassifier/flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
        'train': transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomVerticalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                          [0.229, 0.224, 0.225])
                                    ]),

        'valid': transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])
                                    ]),

        'test': transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])
                                   ]),
    }

    # Load the datasets with ImageFolder
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                     for x in ['train', 'valid', 'test']}

    # Using the image datasets and the trainforms, define the dataloaders
    batch_size = 32
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True)
                   for x in ['train', 'valid', 'test']}

    class_names = image_datasets['train'].classes

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}

    with open('ImageClassifier/cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    in_args = get_input_args()
    device = set_device(in_args.device_opt)
    model = select_model(in_args.arch)
    
    classifier = build_classifier(model,hidden_units=in_args.hidden_units)
    model.classifier = classifier
    model.to(device)
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_args.learning_rate)
    criterion = nn.NLLLoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=4,gamma=0.1,last_epoch=-1)
    epochs = 5
    
    train_classifier(model, criterion, optimizer, scheduler, epochs=in_args.epochs)
    save_checkpoint()

    print("Model trained and checkpoint saved.")

# Call to main function to run the program
if __name__ == "__main__":
    main()
    
# Sources:
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# https://medium.com/datadriveninvestor/creating-a-pytorch-image-classifier-da9db139ba80
# https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad