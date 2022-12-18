from torch import nn
import torch

NUM_CLASSES = 73


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.name = "CNN"

        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=21, stride=1,padding=10)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=11, stride=1,padding=5)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1,padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=5, stride=5)
        self.pool2 = nn.MaxPool1d(kernel_size=5, stride=5)
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.3)
        self.dense1 = nn.Linear(in_features =  20*256, out_features = 256)
        self.dense2 = nn.Linear(in_features = 256, out_features = NUM_CLASSES)

    def forward(self, input):
        x = self.conv1(input)              #x.shape: (batch,64,500) 
        x = x.relu()
        x = self.pool1(x)                   #x.shape: (batch,64,100)
        x = self.drop1(x)
        x = self.conv2(x)                  #x.shape: (batch,128,100)
        x = x.relu()
        x = self.pool2(x)                   #x.shape: (batch,128,20)
        x = self.drop1(x)
        x = self.conv3(x)                  #x.shape: (batch,256,20)
        x = self.drop2(x)
        x = x.relu()
        x = x.flatten(1)           #x.shape: (batch,20*256) #making a 1D vector
        x = self.dense1(x)                #x.shape: (batch, 256)
        x = x.relu()
        x = self.dense2(x)                # x.shape: (batch, 73)
        x = x.sigmoid()
        return x

    def get_parameters(self):
        total_params = 0
        layers_params = {}
        for name, parameter in self.named_parameters():
            params = parameter.numel()
            total_params+=params
            layers_params[name] = params
        
        layers_params['Total Trainable Params'] = total_params
        return layers_params.items()


class CNN_LSTM(nn.Module):

    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.name = "CNN_LSTM"

        self.conv = nn.Conv1d(in_channels=4, out_channels=128, kernel_size=21, stride=1,padding=10)
        self.pool = nn.MaxPool1d(kernel_size=10, stride=10)
        self.drop1 = nn.Dropout(p = 0.2)
        self.dense1 = nn.Linear(in_features = 50, out_features=50)
        ## 2 LSTMs (with 128 temporal recurrent elements, same as max pooling output)
        self.LSTM = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first= True,
                            dropout=0.1, bidirectional=True)
        self.drop2 = nn.Dropout(p = 0.2)
        self.dense2= nn.Linear(in_features =  50*256, out_features = 256)
        self.drop3 = nn.Dropout(p = 0.4)
        self.dense3= nn.Linear(in_features = 256, out_features = NUM_CLASSES)
        
    def forward(self, input):
        x = self.conv(input)                #x.shape: (batch,128,500)
        x = x.relu()
        x = self.pool(x)                    #x.shape: (batch,128,50)
        x = self.drop1(x)
        x = self.dense1(x)                  #x.shape: (batch,128,50)
        x = x.relu()
        x_t = torch.transpose(x, 1, 2)      # x.shape: (batch,128,50)--> x_t.shape: (batch,50,128)
        x, (h_n,h_c) = self.LSTM(x_t)       # x.shape: (batch, 50, 256)  ##need to catch h_n and h_c hidden states even if we dont care about them
        x = x.flatten(1)                     # x.shape: (batch, 50*256=12800)  ##making a 1D vector 
        x = self.drop2(x)
        x = self.dense2(x)                  # x.shape: (batch, 256)
        x = x.relu()
        x = self.drop3(x)
        x = self.dense3(x)                  # x.shape: (batch, 73)
        x = x.sigmoid()

        return x   

    def get_parameters(self):
        total_params = 0
        layers_params = {}
        for name, parameter in self.named_parameters():
            params = parameter.numel()
            total_params+=params
            layers_params[name] = params
        
        layers_params['Total Trainable Params'] = total_params
        return layers_params.items()


class CNN_LSTM2(nn.Module):

    def __init__(self):
        super(CNN_LSTM2, self).__init__()
        self.name = "CNN_LSTM2"

        self.conv = nn.Conv1d(in_channels=4, out_channels=128, kernel_size=21, stride=1,padding=10)
        self.pool = nn.MaxPool1d(kernel_size=10, stride=10)
        self.drop1 = nn.Dropout(p = 0.3)
        self.dense1 = nn.Linear(in_features = 50, out_features=50)
        ## 2 LSTMs (with 128 temporal recurrent elements, same as max pooling output)
        self.LSTM = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first= True,
                            dropout=0.2, bidirectional=True)
        self.drop2 = nn.Dropout(p = 0.3)
        self.dense2= nn.Linear(in_features =  50*256, out_features = 256)
        self.drop3 = nn.Dropout(p = 0.5)
        self.dense3= nn.Linear(in_features = 256*2, out_features = NUM_CLASSES)
        
    def forward(self, *args):
        outputs=[]
        for input in args:
            x = self.conv(input)                #x.shape: (batch,128,500)
            x = x.relu()
            x = self.pool(x)                    #x.shape: (batch,128,50)
            x = self.drop1(x)
            x = self.dense1(x)                  #x.shape: (batch,128,50)
            x = x.relu()
            x_t = torch.transpose(x, 1, 2)      # x.shape: (batch,128,50)--> x_t.shape: (batch,50,128)
            x, (h_n,h_c) = self.LSTM(x_t)       # x.shape: (batch, 50, 256)  ##need to catch h_n and h_c hidden states even if we dont care about them
            x = x.flatten(1)                     # x.shape: (batch, 50*256=12800)  ##making a 1D vector 
            x = self.drop2(x)
            x = self.dense2(x)                  # x.shape: (batch, 256)
            x = x.relu()
            x = self.drop3(x)
            outputs.append(x)

        x = torch.cat((outputs[0],outputs[-1]),dim=1)   # x.shape: (batch, 512)        
        x = self.dense3(x)                              # x.shape: (batch, 73)
        x = x.sigmoid()

        return x   

    def get_parameters(self):
        total_params = 0
        layers_params = {}
        for name, parameter in self.named_parameters():
            params = parameter.numel()
            total_params+=params
            layers_params[name] = params
        
        layers_params['Total Trainable Params'] = total_params
        return layers_params.items()