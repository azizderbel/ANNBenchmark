import torch
import torch.nn as nn
from dataset import train_loader,test_loader

import model.mlp as mlp
import model.cnn as cnn
import model.rnn as rnn
import model.gru as gru
import model.lstm as lstm
import matplotlib.pyplot as plt
import graphs as gr

accuracy_dict = {}
n_epochs = 20  #20
learning_rate = 0.03
loss = torch.nn.CrossEntropyLoss() 

mlp_model = mlp.MlpClassifier(input_size=1*28*28,out_size=10)
cnn_model = cnn.CnnClasifier()
rnn_model = rnn.RnnClasifier(input_size=28,hidden_size=128,n_classes=10,n_layer=1)
gru_model = gru.GruClasifier(input_size=28,hidden_size=128,n_classes=10,n_layer=1)
lstm_model = lstm.LstmClasifier(input_size=28,hidden_size=128,n_classes=10,n_layer=1)

mlp_optimizer = torch.optim.SGD(mlp_model.parameters(),lr=learning_rate)
cnn_optimizer = torch.optim.SGD(cnn_model.parameters(),lr=learning_rate)
rnn_optimizer = torch.optim.SGD(rnn_model.parameters(),lr=learning_rate)
gru_optimizer = torch.optim.SGD(gru_model.parameters(),lr=learning_rate)
lstm_optimizer = torch.optim.SGD(lstm_model.parameters(),lr=learning_rate)


mlp.train_model(mlp_model,train_loader=train_loader,loss=loss,optimizer=mlp_optimizer,epochs=n_epochs)
cnn.train_model(cnn_model,train_loader=train_loader,loss=loss,optimizer=cnn_optimizer,epochs=n_epochs)
rnn.train_model(rnn_model,train_loader=train_loader,loss=loss,optimizer=rnn_optimizer,epochs=n_epochs)
gru.train_model(gru_model,train_loader=train_loader,loss=loss,optimizer=gru_optimizer,epochs=n_epochs)
lstm.train_model(lstm_model,train_loader=train_loader,loss=loss,optimizer=lstm_optimizer,epochs=n_epochs)

accuracy_dict['MLP'] = mlp.score(mlp_model,test_loader=test_loader) # 87.5
accuracy_dict['CNN'] = cnn.score(cnn_model,test_loader=test_loader) # 98.35
accuracy_dict['RNN'] = rnn.score(rnn_model,test_loader=test_loader) # 97.26
accuracy_dict['GRU'] = gru.score(gru_model,test_loader=test_loader) # 97.25
accuracy_dict['LSTM'] = lstm.score(lstm_model,test_loader=test_loader) # 97.30

gr.plotAccuracy(accuracy_dict)

#plt.legend()
plt.show()




