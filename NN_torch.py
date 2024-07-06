import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms


class NN(nn.Module):
  def __init__(self,ni,no,nh_list,nonlinearity='relu',nonlinearity_out='relu'):  
    super().__init__()

    self.nonlinearity = nonlinearity
    self.nonlinearity_out = nonlinearity_out
    self.input=ni
    self.num_hidden_layers = len(nh_list)

    self.layers=nh_list
    self.layers.insert(0, ni)
    self.layers.append(no)
     
    self.dense_layers = nn.ModuleList()
    for i in range(self.num_hidden_layers):
      self.dense_layers.append(nn.Linear(self.layers[i], self.layers[i + 1])) #e.g., 128 --> 10; 10 --> 5

    self.output_layer=nn.Linear(self.layers[-2], self.layers[-1])             #e.g., 5 --> 1

  def forward(self,x):
    x=x.view(-1,self.input)
    for layer in self.dense_layers:
      if self.nonlinearity == 'relu':
        x = torch.relu(layer(x))
      elif self.nonlinearity is None:
        x = layer(x)
    
    if self.nonlinearity_out == 'relu':
      out = torch.relu(self.output_layer(x))
    elif self.nonlinearity_out is None:
      out = self.output_layer(x)

    return out
  
# Example Use: 
# NN_net=NN(ni=10,no=2,nh_list=[10,5],nonlinearity_out=None).float()                                              
# x_input = torch.tensor(X_train).float()                                    
# y_input =  torch.tensor(y_train).float()          

# num_epochs=10000
# lr=1e-3                                                        

# loss_fn=nn.MSELoss()
# optimizer=optim.Adam(NN_net.parameters(),lr=lr)  

# ls_loss = []

# for i in range(num_epochs):
#   total_loss=0
#   y_pred=NN_net(x_input)                                             
#   loss=loss_fn(y_pred,y_input)
#   total_loss+=loss.item()
  
#   optimizer.zero_grad()
#   loss.backward()
#   optimizer.step()

#   ls_loss.append(total_loss) 

#   print("epoch:", i, "loss:",total_loss)
#   print("__________________________________________________________")