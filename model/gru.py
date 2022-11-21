import torch 
import torch.nn as nn
import numpy as np
import graphs as gr


class GruClasifier(nn.Module):
    def __init__(self,input_size,hidden_size,n_classes,n_layer) -> None:
        super(GruClasifier,self).__init__()

        self.n_layers = n_layer
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size,hidden_size,n_layer,batch_first=True) # expect an input of [batch,sequence,input_size] 
                                                                           # output = [batch_size,seq length,hidden_state]
        self.fc1 = nn.Linear(hidden_size,out_features=n_classes)

    def forward(self,x):
        h0 = torch.zeros(self.n_layers,x.shape[0],self.hidden_size) # [1,28,128] [layer,seq,hidden]
        out,_ = self.gru(x,h0) # an out with shape of [100,28,128]
        out = out[:,-1,:] # [100,128] [batch,hidden_size]
        return self.fc1(out) 


def train_model(model,train_loader,loss,optimizer,epochs = 100):
    loss_counter = 0
    loss_step = 100
    all_losses = []
    for epoch in range(epochs):
        for i,(feature,labels) in enumerate(train_loader):
            images = feature.reshape(-1,28,28)
            output = model(images)

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
    gr.plotLoss(all_losses,'GRU loss',epochs)

def score(model,test_loader):
    n_correct_answers = 0
    n_samples = 0
    with torch.no_grad():
        for i,(features,labels) in enumerate(test_loader):
            n_samples += labels.shape[0]
            images = features.reshape(-1,28,28)
            labels_pred = model(images) # the output shape [100,10]
            values,indices = torch.max(labels_pred,1) # return a named tuple of axis maximums (value,indices)
            # values [batch_size] # indices [batch_size]
            equivalence = (indices == labels) # boolean tensor [batch_size] 
            n_correct_answers += equivalence.sum().item() # how many true answers there are out there
        accuracy = 100 * (n_correct_answers/n_samples)
        print(f'the model accuracy is {accuracy}')
        return accuracy