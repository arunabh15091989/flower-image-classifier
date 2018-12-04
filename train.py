import utility
import argparse
ap = argparse.ArgumentParser(description='train.py')
# Command Line ardguments

ap.add_argument('data_dir', nargs='*', action="store", default='/home/workspace/paind-project/flowers/')
ap.add_argument('--gpu', dest="gpu", action="store", default=True)
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.003)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=3)
ap.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)

pa = ap.parse_args()
where = pa.data_dir
path = pa.save_dir
lr = pa.learning_rate
structure = pa.arch
dropout = pa.dropout
hidden_layer = pa.hidden_units
gpu = pa.gpu
epochs = pa.epochs

trainloader, validloader, testloader, train_dataset, valid_dataset,test_dataset,  = utility.create_loaders(where)

model, optimizer, criterion, input_size = utility.nn_setup(structure,dropout,hidden_layer,lr,gpu)

utility.train_network(model, optimizer, criterion, epochs, trainloader, validloader, gpu)
utility.test_accuracy(model,testloader,criterion,gpu)
utility.save_model_checkpoint(model,input_size,epochs,'/home/workspace/paind-project/',structure,lr,train_dataset,optimizer,102)

print('your model is trained')