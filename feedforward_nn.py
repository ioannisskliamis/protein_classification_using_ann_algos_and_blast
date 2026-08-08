import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNN(nn.Module):
    def __init__(self, input_size, classes, layers, nodes):
        super().__init__()
        self.layers = nn.ModuleList()                                     #Create list of layers according to number of layers the user has given 
        in_features = input_size
        for layer_num in range(layers):
            if layer_num == layers - 1:
                out_features = classes
            else:
                out_features = nodes
            self.layers.append(nn.Linear(in_features, out_features))
            in_features = out_features

        self.dropout = nn.Dropout(0.2)                                    #Add dropout

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i, layer in enumerate(self.layers):
            if i != len(self.layers) - 1:
                x = F.relu(layer(x))                                      #Use of ReLu activation function after every layer
                x = self.dropout(x)
        x = self.layers[-1](x)


        return x
