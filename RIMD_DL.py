# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:49:21 2020

@author: Wamiq

"""
from sklearn.preprocessing import Normalizer
import torchvision.transforms as transforms
from torchvision import datasets
import pickle as pkl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy
from sklearn.metrics import confusion_matrix
import itertools


"""
    # loadv data from Pickle file with this function
    # pickleFileName = name of pickle file that needed to load
    # return dataArray=loaded data from pickle file
"""
def loadPickle(pickleFileName):
    with open(pickleFileName, 'rb') as fileObject:
        dataArray = pkl.load(fileObject)
    fileObject.close()
    return dataArray


'-------------------------------------------------------------------------------------'
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'-------------------------------------------------------------------------------------'
# Hyper parameters
num_epochs = 250
num_classes = 25
batch_size = 32
learning_rate = 0.01
n_params=[16,32,64]
'-------------------------------------------------------------------------------------'

"""
    Reading dataset from pickle file
"""

X_train = loadPickle("XTrainPickle.pkl")
y_train = loadPickle("yTrainPickle.pkl")

X_validate = loadPickle("XValidatePickle.pkl")
y_validate = loadPickle("yValidatePickle.pkl")

X_test = loadPickle("XTestPickle.pkl")
y_test = loadPickle("yTestPickle.pkl")

"""
    performing L2 normalization
"""
Data_normalizer = Normalizer(norm='l2').fit(X_train)
X_train = Data_normalizer.transform(X_train)

Data_normalizer = Normalizer(norm='l2').fit(X_validate)
X_validate = Data_normalizer.transform(X_validate)

Data_normalizer = Normalizer(norm='l2').fit(X_test)
X_test = Data_normalizer.transform(X_test)


"""
    # Transformation
"""
transform = transforms.Compose([
                                transforms.Resize((32,32)),
                                transforms.ToTensor()                         
                                ])
#transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])//l2 normalization
#train_dataset = datasets.ImageFolder(root=train_path,transform=transform)
#val_dataset = datasets.ImageFolder(root=val_path,transform=transform)  
#test_dataset=datasets.ImageFolder(root=test_path,transform=transform)



'-------------------------------------------------------------------------------------'
"""
# Data loader
#train_loader = torch.utils.data.DataLoader(dataset=X_train,
                                            #batch_size=batch_size, 
                                            #shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=X_validate,
                                          batch_size=batch_size, 
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=X_test,
                                          batch_size=batch_size, 
                                          shuffle=True)
"""
'-------------------------------------------------------------------------------------'

'-------------------------------------------------------------------------------------'
# Convolutional neural network
class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels =1, out_channels =1, kernel_size=1, stride=1, padding=0),   
            #nn.BatchNorm1d(1),
            #nn.Softmax(),
            #nn.ReLU(),  //softmax activation function
            nn.MaxPool1d(kernel_size=1, stride=1))
        #self.lstm = nn.LSTM(1024,256)
        self.fc = nn.Linear(1024, num_classes)

      
    def forward(self, x):
        out = self.layer1(x)
        #out = self.lstm(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = ConvNet(num_classes).float()#.to(device)
'-------------------------------------------------------------------------------------'
'-------------------------------------------------------------------------------------'
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

'-------------------------------------------------------------------------------------'

# Train the model
confusion_mat_train = torch.zeros(num_classes, num_classes)
confusion_mat_val = torch.zeros(num_classes, num_classes)

for epoch in range(num_epochs):
    total=0
    correct = 0
    train_acc=0
    train_hist=[]
    images= torch.from_numpy(numpy.array(X_train))
    labels=torch.from_numpy(numpy.array(y_train))
    images = images.reshape(6537,1,1024)
    outputs= model(images.float())
    loss_train = criterion(outputs, labels)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    train_hist.append(loss_train)
        
    # Backward and optimize
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    for t, p in zip(labels.view(-1), predicted.view(-1)):
        confusion_mat_train[t.long(), p.long()] += 1

    train_acc=(correct/total)*100
    print ('Epoch [{}/{}], Train Loss: {:.4f}, train Acc: {:,.4f}'
                   .format(epoch+1, num_epochs, loss_train.item(), train_acc))

    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        val_acc=0
        val_hist=[]
        images= torch.from_numpy(numpy.array(X_validate))
        labels=torch.from_numpy(numpy.array(y_validate))
        images = images.reshape(2802,1,1024)
        outputs = model(images.float())
        loss_val = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        val_hist.append(loss_val)
        for t, p in zip(labels.view(-1), predicted.view(-1)):
            confusion_mat_val[t.long(), p.long()] += 1
        val_acc=(correct/total)*100
        print ('Epoch [{}/{}], Val Loss: {:.4f} , Val Acc: {:,.4f}'
                   .format(epoch+1, num_epochs, loss_val.item(), val_acc))
        
'-----------------------------------------------------------------------------'
#Plotting graph        
plt.plot(train_hist,label='Training loss')
plt.plot(val_hist,label='Validation loss')
plt.legend()
plt.show()



'-----------------------------------------------------------------------------'
#Printingconfusion matrix
print('Confusion Matrix for Training Data', confusion_mat_train)
print('Confusion Matrix for Validation Data', confusion_mat_val)

'-----------------------------------------------------------------------------'       
### Model save ####
torch.save(model.state_dict(), 'E:/RIS/CNN.pt')



'-----------------------------------------------------------------------------'
# Test the model###
#model = ConvNet(num_classes).to(device)
model.load_state_dict(torch.load('E:/RIS/CNN.pt'))
model.eval()  
confusion_mat = torch.zeros(num_classes, num_classes)
with torch.no_grad():
    correct = 0
    total = 0
    images= torch.from_numpy(numpy.array(X_test))
    labels=torch.from_numpy(numpy.array(y_test))
    outputs = model(images.float())
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    for t, p in zip(labels.view(-1), predicted.view(-1)):
        confusion_mat[t.long(), p.long()] += 1

    print('Test Accuracy of the model on the 20 test images: {} %'.format(100 * correct / total))
    print('Confusion Matrix for Test Data',confusion_mat)
