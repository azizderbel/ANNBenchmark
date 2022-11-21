import torch
import torch.nn as nn
import graphs as gr
import numpy as np

class CnnClasifier(nn.Module):

    def __init__(self) -> None:
        super(CnnClasifier,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=5,kernel_size=3,stride=1) # the ouput is [100,5,26,26]
        self.max_pool = nn.MaxPool2d(kernel_size=2,stride=2) # resize feature maps to the half [100,5,13,13]
        self.conv2 = nn.Conv2d(in_channels=5,out_channels=10,kernel_size=3,stride=1) # the ouput is [100,10,11,11]
        # flatten the input
        self.fc1 = nn.Linear(in_features=10*5*5,out_features=84) # output [100,84]
        self.fc2 = nn.Linear(in_features=84,out_features=10) # output [100,10]
    

    def forward(self,x):  # expects [100,1,28,28]
        out = nn.functional.relu(self.conv1(x))
        out = self.max_pool(out)
        out = nn.functional.relu(self.conv2(out))
        out = self.max_pool(out)
        out = out.reshape(-1,10*5*5) # [100,250]
        out = self.fc1(out)
        out = self.fc2(out)
        return out

def train_model(model,train_loader,loss,optimizer,epochs = 100):
    loss_counter = 0
    loss_step = 100
    all_losses = []
    for epoch in range(epochs):
        for i,(features,labels) in enumerate(train_loader):
                # forward pass
            output = model(features)
                # calculate the loss
            l = loss(output,labels)
            loss_counter += l.item()
                # backward pass
            optimizer.zero_grad()
            l.backward()
                # update waights
            optimizer.step()
            
            if i % loss_step  == 0:
                print(f'epoch : {epoch+1}/{epochs} , loss :{l.item():.4f}')
                all_losses = np.append(all_losses,loss_counter/loss_step)
                loss_counter = 0
    gr.plotLoss(all_losses,'CNN loss',epochs)

def score(model,test_loader):
    n_correct_answers = 0
    n_samples = 0
    with torch.no_grad():
        for i,(features,labels) in enumerate(test_loader):
            n_samples += labels.shape[0]
            labels_pred = model(features) # the output shape [100,10]
            values,indices = torch.max(labels_pred,1) # return a named tuple of axis maximums (value,indices)
            # values [batch_size] # indices [batch_size]
            equivalence = (indices == labels) # boolean tensor [batch_size] 
            n_correct_answers += equivalence.sum().item() # how many true answers there are out there
        accuracy = 100 * (n_correct_answers/n_samples)
        print(f'the model accuracy is {accuracy}')
        return accuracy