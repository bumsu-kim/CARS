import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from torch.optim.lr_scheduler import StepLR

# This block of code fetches the data, and defines a function that
# splits the data into test/train, and into batches.
# Note that this function will only download the data once. Subsequent 
# calls will load the data from the hard drive

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch 



def MNIST_Loaders(train_batch_size, test_batch_size=None):
    if test_batch_size is None:
        test_batch_size = train_batch_size
    
    normalize = transforms.Normalize((0.,), (1.,))
#     normalize = transforms.Normalize((0.1307,), (0.3081,))
    Clean = transforms.Compose([transforms.ToTensor(), normalize])
   
    #!wget www.di.ens.fr/~lelarge/MNIST.tar.gz
    #!tar -zxvf MNIST.tar.gz
    
    train_data = datasets.MNIST('./', train=True,
                                   download=True, transform=Clean)
    test_data = datasets.MNIST('./', train=False,
                                  download=True, transform=Clean)
    
    train_loader = torch.utils.data.DataLoader(train_data,
                    batch_size=train_batch_size)
    
    test_loader = torch.utils.data.DataLoader(test_data,
                    batch_size=test_batch_size)
    
    return train_loader, test_loader
    
# This block of code sets up the network. We'll use the LeNet-5
# architecture.

input_size = [28,28]

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6,
                               kernel_size=5 , stride=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,
                              kernel_size=5, stride=1) 
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120,
                              kernel_size=4, stride=1)
        self.fc_1 = nn.Linear(in_features=120,out_features=84)  
        self.fc_2 = nn.Linear(in_features=84, out_features=10)
            
    def forward(self,u):
        u = self.conv1(u)  # apply first convolutional layer
        u = self.relu(u)   # apply ReLU activation
        u = self.pool(u)   # apply max-pooling
        u = self.conv2(u)  # apply second convolutional layer
        u = self.relu(u)   # Apply ReLU activation
        u = self.pool(u)
        u = self.conv3(u)  # Apply third and final convolutional layer
        u = torch.flatten(u, 1)
        u = self.fc_1(u)
        u = self.relu(u)
        u = self.fc_2(u)
        u = self.relu(u)
        y = F.log_softmax(u, dim=1)
        return y
	
# Some useful functions to keep tabs on the training, courtesy of Samy Wu Fung

def get_stats(net, test_loader):
        test_loss=0
        correct=0
        with torch.no_grad():
            for d_test, labels in test_loader:
                batch_size = d_test.shape[0]
                y = net(d_test)  # apply the network to the test data
                test_loss += batch_size*F.nll_loss(y, labels).item() # sum up batch loss
                
                pred = y.argmax(dim=1,keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
                
        test_loss /= len(test_loader.dataset)
        test_acc = 100.*correct/len(test_loader.dataset)
        
        return test_loss, test_acc, correct
                
        
def model_params(net):
    table = PrettyTable(["Network Component", "# Parameters"])
    num_params = 0
    for name, parameter in net.named_parameters():
        if not parameter.requires_grad:
            continue
        table.add_row([name, parameter.numel()])
        num_params += parameter.numel()
    table.add_row(['Total', num_params])
    return table

# The training function

def train_net(net, num_epochs, train_loader, test_loader, optimizer,
              checkpt_path):
    
    loss_ave = 0.0
    train_acc = 0.0
    best_test_acc = 0.0
    
    test_loss_hist = []
    test_acc_hist = []
    
    print(net)
    print(model_params(net))
    print('\nTraining Network')
    
    # initialize a learning rate scheduler
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(num_epochs):
        tot = len(train_loader)
        
        for idx, (data, labels) in enumerate(train_loader):
            batch_size = data.shape[0]
            
            optimizer.zero_grad()
            
            # forward and backward then take a step of optimizer
            y = net(data)
            loss = F.nll_loss(y, labels)
            loss.backward()
            optimizer.step()
            
        # Output some training stats
        if (epoch+1) % 1 == 0:
            test_loss, test_acc, correct = get_stats(net, test_loader)
            print('Number of epochs= {:03d} Test loss = {:.3f} and Test accuracy = {:.3f}'.format(epoch+1, test_loss, test_acc))
            
        # Save weights every ten epochs
        if (epoch +1) % 10 == 0 and test_acc > best_test_acc:
            best_test_acc = test_acc
            state = {
                'test_loss_hist': test_loss_hist,
                'test_acc_hist': test_acc_hist,
                'net_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            file_name = checkpt_path + 'MNIST_weights.pth'
            torch.save(state, file_name)
            print('Model weights saved to ' + file_name)
            
        # advance the learning rate schedule
        scheduler.step()
    return net
        
# Initialize model and prepare for training
load_weights = False

model = Net()
max_epochs=100 # train for max_epochs full passes over the data
learning_rate=1.0
optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
checkpt_path = './models/'  # Best practice is to periodically save, or checkpoint the weights
batch_size = 128
save_dir = './'

if load_weights:
    state = torch.load('modelsMNIST_weights.pth')
    model.load_state_dict(state['net_state_dict'])

train_loader, test_loader = MNIST_Loaders(batch_size)
model = train_net(model, max_epochs, train_loader, test_loader,
                 optimizer, checkpt_path)

   
