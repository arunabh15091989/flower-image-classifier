import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import copy
from collections import OrderedDict

arch = {"vgg16":25088,
        "vgg19":25088,
        "densenet121":1024,
        "alexnet":9216}


def create_loaders(data_dir):
    print('data dir is \n')
    print(data_dir)
    train_dir = data_dir + 'train'
    valid_dir = data_dir + 'valid'
    test_dir = data_dir + 'test'
    
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ]),
    }
    # Just normalization for validation and test
    # Data augmentation and normalization for training
    dirs = {'train': train_dir, 'valid': valid_dir, 'test' : test_dir}
    image_datasets = {x: datasets.ImageFolder(dirs[x],   transform=data_transforms[x]) for x in ['train', 'valid', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['train', 'valid', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
    class_names = image_datasets['train'].classes
    return (dataloaders['train'],dataloaders['valid'],dataloaders['test'],image_datasets['train'],image_datasets['valid'],image_datasets['test'])
     
    # return          (dataloaders['train'],dataloaders['valid'],dataloaders['test'],dataset_sizes['train'],dataset_sizes['valid'],dataset_sizes['test'],class_names)

def nn_setup(structure='densenet121',dropout=0.5, hidden_layer = 120,lr = 0.003,gpu=True):
    '''
    Arguments: The architecture for the network(alexnet,densenet121,vgg16,vgg19), the hyperparameters for the network (hidden layer 1 nodes, dropout and learning rate) and whether to use gpu or not
    Returns: The set up model, along with the criterion and the optimizer fo the Training
    '''
    input_size=0
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = arch['vgg16']
    elif structure == 'vgg19':
        model = models.vgg19(pretrained=True)
        input_size = arch['vgg19']
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = arch['densenet121']
    elif structure == 'alexnet':
        model = models.alexnet(pretrained = True)
        input_size = arch['alexnet']
    else:
        print("Im sorry but {} is not a valid model.Did you mean vgg16,densenet121,or alexnet?".format(structure))
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088,hidden_layer)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_layer, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier
    learning_rate = 0.003
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    if torch.cuda.is_available() and gpu:
        model.cuda()
    elif gpu and torch.cuda.is_available() == False:
        print('GPU processing selected but no NVIDIA drivers found... Training under CPU.')
    return model,optimizer,criterion,input_size

def train_network(model, optimizer, criterion, epochs, trainloader, validloader,gpu):
    '''
    Arguments: The model, the criterion, the optimizer, the number of epochs, the dataset, and whether to use a gpu or not
    Returns: Nothing
    This function trains the model over a certain number of epochs and displays the training,validation and accuracy every "print_every" step     using cuda if specified. The training method is specified by the criterion and the optimizer which are NLLLoss and Adam respectively
    '''
    print_every = 40
    steps=0
    model.to('cuda')
    
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            if torch.cuda.is_available() and gpu:
                inputs.to('cuda')
                labels.to('cuda')
            if gpu and torch.cuda.is_available() == False:
                print('GPU processing selected but no NVIDIA drivers found... Training under CPU.')
            steps += 1
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                model.eval()    
                validation_loss = 0
                accuracy=0

                for ii, (inputs2,labels2) in enumerate(validloader):
                    optimizer.zero_grad()
                    inputs2 = inputs2.to('cuda')
                    labels2 = labels2.to('cuda')
                    with torch.no_grad():    
                        outputs2 = model.forward(inputs2)
                        validation_loss = criterion(outputs2,labels2)
                        ps = torch.exp(outputs2).data
                        equality = (labels2.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()

                validation_loss = validation_loss / len(validloader)
                accuracy = accuracy /len(validloader)
                
                print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Training Loss: {:.4f}".format(running_loss/print_every),
                  "Validation Loss {:.4f}".format(validation_loss),
                  "Validation Accuracy: {:.4f}".format(accuracy))
                running_loss = 0
    print("-------------- Finished training -----------------------")

def test_accuracy(model, testloader, criterion, gpu):    
    model.to('cuda')
    model.eval()       
    test_acc = 0
    test_loss = 0
    for ii, (inputs3,labels3) in enumerate(testloader):
        inputs3 = inputs3.to('cuda')
        labels3 = labels3.to('cuda')
        with torch.no_grad():
            outputs3 = model.forward(inputs3)
            test_loss = criterion(outputs3,labels3)
            ps = torch.exp(outputs3).data
            equality = (labels3.data == ps.max(1)[1])
            test_acc += equality.type_as(torch.FloatTensor()).mean()
    test_loss = test_loss / len(testloader)   
    test_acc = test_acc/len(testloader)
    print("Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
          "Test Accuracy: {:.3f}".format(test_acc/len(testloader)))
    pass


def save_model_checkpoint(model, input_size, epochs, save_dir, arch, learning_rate, train_dataset, optimizer, output_size):
    """
    Saves the trained and tested module by outputting a checkpoint file.
    Parameters:
        model - Previously trained and tested CNN
        input_size - Input size used on the specific CNN
        epochs - Number of epochs used to train the CNN
        save_dir - Directory to save the checkpoint file(default- current path)
        arch - pass string value of architecture used for loading
    Returns:
        None - Use module to output checkpoint file to desired directory
    """
    checkpoint = {
    'input_size':input_size,
    'epochs':epochs,
    'arch':arch,
    'hidden_units':[each.out_features for each in model.classifier if hasattr(each, 'out_features') == True],
    'output_size': output_size,
    'learning_rate': learning_rate,
    'class_to_idx': train_dataset.class_to_idx,
    'optimizer_dict': optimizer.state_dict(),
    'classifier': model.classifier,
    'state_dict': model.state_dict() 
    }
    model.class_idx =train_dataset.class_to_idx
    #Save checkpoint in current directory unless otherwise specified by save_dir
    if len(save_dir) == 0:
        save_path = save_dir + 'checkpoint.pth'
    else:
        save_path = save_dir + '/checkpoint.pth'
    torch.save(checkpoint, save_path)
    print('Model saved at {}'.format(save_path))
    pass

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    arch = checkpoint['arch']
    if arch == 'vgg16':
        load_model = models.vgg16(pretrained=True)
    elif arch == 'vgg19':
        load_model = models.vgg19(pretrained=True)
    elif arch == 'densenet':
        load_model = models.densenet121(pretrained=True)
    else:
        print('error: arch not recognized. please use either vgg16, vgg19 or densenet121 as arch parameter')
    for param in load_model.parameters():
        param.requires_grad = False   
    load_model.classifier = checkpoint['classifier']
    load_model.load_state_dict(checkpoint['state_dict'])
    return (load_model, arch, checkpoint['class_to_idx'])



def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean  
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1) 
    ax.imshow(image)
    return ax

