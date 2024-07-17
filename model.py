import torch
import torch.nn as nn

#Define the model for the neural network

class SLP_Model(nn.Module):

  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(3250, 64) # Input is feature data with 13 columns
    self.batch_norm1 = nn.BatchNorm1d(64) 
    self.av1 = nn.ReLU()
    self.drop1 = nn.Dropout(p=0.1)
    self.fc2 = nn.Linear(64, 64) # Hidden layer
    self.batch_norm2 = nn.BatchNorm1d(64) 
    self.av2 = nn.ReLU()
    self.drop2 = nn.Dropout(p=0.2)
    self.fc3 = nn.Linear(64, 64) # Hidden layer
    self.batch_norm3 = nn.BatchNorm1d(64) 
    self.av3 = nn.ReLU()
    self.drop3 = nn.Dropout(p=0.2)
    self.fc4 = nn.Linear(64, 7) # Output layer is classifiction
    self.avout = nn.LogSoftmax(dim=-1)
    return

  def forward(self, x):
    x = self.fc1(x)
    x = self.batch_norm1(x)
    x = self.av1(x)
    # x = self.drop1(x)
    x = self.fc2(x)
    x = self.batch_norm2(x)
    x = self.av2(x)
    # x = self.drop2(x)
    x = self.fc3(x)
    x = self.batch_norm3(x)
    x = self.av3(x)
    # x = self.drop3(x)    
    x = self.fc4(x)
    x = self.avout(x)
    return x
