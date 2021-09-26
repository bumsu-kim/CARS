# coding: utf-8
# This block of code fetches the data, and defines a function that
# splits the data into test/train, and into batches.
# Note that this function will only download the data once. Subsequent 
# calls will load the data from the hard drive

def MNIST_Loaders(train_batch_size, test_batch_size=None):
    if test_batch_size is None:
        test_batch_size = train_batch_size
    
    normalize = transforms.Normalize((0.1307,), (0.3081,))
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
    
