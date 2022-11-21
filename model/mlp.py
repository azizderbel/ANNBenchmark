import torch 
import torch.nn as nn
import numpy as np
import graphs as gr


class MlpClassifier(nn.Module):

    def __init__(self,input_size,out_size) -> None:
        super(MlpClassifier,self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(in_features=input_size,out_features=200)
        self.fc2 = nn.Linear(in_features=200,out_features=120)
        self.fc3 = nn.Linear(in_features=120,out_features=60)
        self.fc4 = nn.Linear(in_features=60,out_features=out_size)


    def forward(self,x):
        out = nn.functional.relu(self.fc1(x))
        out = nn.functional.relu(self.fc2(out))
        out = nn.functional.relu(self.fc3(out))
        out = nn.functional.relu(self.fc4(out))
        return out

def train_model(model,train_loader,loss,optimizer,epochs = 100):
    loss_counter = 0
    loss_step = 100
    all_losses = []
    for epoch in range(epochs):
        for i,(features,labels) in enumerate(train_loader):
                # forward pass
            images = features.reshape(-1,28*28)
            output = model(images)
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
    gr.plotLoss(all_losses,'Feed forward NN loss',epochs)

def score(model,test_loader):
    n_correct_answers = 0
    n_samples = 0
    with torch.no_grad():
        for i,(features,labels) in enumerate(test_loader):
            images = features.reshape(-1,28*28)
            n_samples += labels.shape[0]
            labels_pred = model(images) # the output shape [100,10]
            values,indices = torch.max(labels_pred,1) # return a named tuple of axis maximums (value,indices)
            # values [batch_size] # indices [batch_size]
            equivalence = (indices == labels) # boolean tensor [batch_size] 
            n_correct_answers += equivalence.sum().item() # how many true answers
        accuracy = 100 * (n_correct_answers/n_samples)
        print(f'the model accuracy is {accuracy}')
        return accuracy
        




