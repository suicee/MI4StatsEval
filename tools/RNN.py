from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from .FCN import ANN_Classifier
class LSTM_regressor(torch.nn.Module):
    def __init__(self, seq_len, feature_size ,hidden_dim,inter_dim, output_dim, num_layers=1,dropout=0.2, bidirectional=False):
        super(LSTM_regressor, self).__init__()
        self.seq_len = seq_len
        self.feature_size=feature_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        
        self.rnn = nn.LSTM(feature_size,hidden_dim, num_layers)
        self.fc1 = torch.nn.Linear(hidden_dim, inter_dim)
        self.fc2= torch.nn.Linear(inter_dim,output_dim)
        
#         self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        

        output, (hn, cn) = self.rnn(x)
        # o=torch.cat((hn[-1],cn[-1]),dim=1)
        o=self.fc1(hn[-1])
        o=F.relu(o)
        y_pred=self.fc2(o)

        # y_pred = self.fc(hn[-1])
        
        return y_pred


class LSTM_regressor_2(torch.nn.Module):
    def __init__(self, seq_len, feature_size ,hidden_dim,output_dim,fc_dims:list, num_layers=1,apply_dropout=False,dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.feature_size=feature_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # self.batch_size = batch_size
        # self.bidirectional = bidirectional
        self.num_layers = num_layers
        
        self.rnn = nn.LSTM(feature_size,hidden_dim, num_layers,batch_first=True)

        self.fc_input_layer=nn.Linear(hidden_dim,fc_dims[0])
        
        fc_modules=[]
        
        for i in range(1,len(fc_dims)):
            fc_modules.append(nn.ReLU())
            if apply_dropout:
                fc_modules.append(nn.Dropout(dropout))
            fc_modules.append(nn.Linear(fc_dims[i-1],fc_dims[i]))
        fc_modules.append(nn.ReLU())


        self.fc_sequence=nn.Sequential(*fc_modules)
        self.output=nn.Linear(fc_dims[-1],output_dim)
        
#         self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        

        output, (hn, cn) = self.rnn(x)
        # o=torch.cat((hn[-1],cn[-1]),dim=1)
        fc_input=self.fc_input_layer(hn[-1])
        fc_output=self.fc_sequence(fc_input)
        output=self.output(fc_output)
        
        return output





class LSTM_Classifier(torch.nn.Module):
    def __init__(self, seq_len, feature_size ,RNN_hidden_dim,param_dim,fc_dims:list, num_layers=1,apply_dropout=False,dropout=0.2):
        super(LSTM_Classifier, self).__init__()
        self.seq_len = seq_len
        self.feature_size=feature_size
        self.hidden_dim = RNN_hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        # if apply_dropout:
        
        #     self.rnn = nn.LSTM(feature_size,RNN_hidden_dim, num_layers,batch_first=True,dropout=dropout)
        # else:
        #     self.rnn = nn.LSTM(feature_size,RNN_hidden_dim, num_layers,batch_first=True)
        self.rnn = nn.LSTM(feature_size,RNN_hidden_dim, num_layers,batch_first=True)
        self.fc_input_layer=nn.Linear(RNN_hidden_dim+param_dim,fc_dims[0])
        
        fc_modules=[]
        
        for i in range(1,len(fc_dims)):
            fc_modules.append(nn.ReLU())
            # fc_modules.append(nn.ELU())
            if apply_dropout:
                fc_modules.append(nn.Dropout(dropout))
            fc_modules.append(nn.Linear(fc_dims[i-1],fc_dims[i]))
        fc_modules.append(nn.ReLU())
        # fc_modules.append(nn.ELU())


        self.fc_sequence=nn.Sequential(*fc_modules)
        self.output=nn.Linear(fc_dims[-1],1)
        


    def forward(self, x,params):
        

        output, (hn, cn) = self.rnn(x)

        fc_input=self.fc_input_layer(torch.cat((hn[-1],params),dim=1))
        fc_output=self.fc_sequence(fc_input)
        output=torch.sigmoid(self.output(fc_output))
 
        return output



class Res_ANN(torch.nn.Module,):
    '''
    Fully connected NN with res units, all layers should have the same dimension.
    '''
    def __init__(self,n_layers=3,layer_dim=256,apply_dropout=True,dropout=0.2) -> None:
        super().__init__()

        fc_modules=nn.ModuleList()
        
        for i in range(n_layers):
            if apply_dropout:
                fc_modules.append(nn.Dropout(dropout))
            fc_modules.append(nn.Linear(layer_dim,layer_dim))


        self.fc_sequence=fc_modules
        


    def forward(self,x):

        for module in self.fc_sequence:
            x=x+module(x)
            x=torch.relu(x)

        return x



class linear_dropout_layer(torch.nn.Module,):
    '''
    linear+dropout
    '''
    def __init__(self,input_dim,output_dim,droput=0.2) -> None:
        super().__init__()

        self.fc=nn.Linear(input_dim,output_dim)
        self.dp=nn.Dropout(droput)
        


    def forward(self,x):

        return self.dp(self.fc(x))

class Res_ANN_new(torch.nn.Module,):
    '''
    Fully connected NN with res units, all layers should have the same dimension.
    '''
    def __init__(self,n_layers=3,layer_dim=256,apply_dropout=True,dropout=0.2) -> None:
        super().__init__()

        fc_modules=nn.ModuleList()
        
        for i in range(n_layers):
            if apply_dropout:
                fc_modules.append(linear_dropout_layer(layer_dim,layer_dim,droput=dropout))
            else:
                fc_modules.append(nn.Linear(layer_dim,layer_dim))


        self.fc_sequence=fc_modules
        


    def forward(self,x):

        for module in self.fc_sequence:
            x=x+module(x)
            x=torch.relu(x)

        return x

class LSTM_Classifier_Res(torch.nn.Module):
    def __init__(self, seq_len, feature_size ,RNN_hidden_dim,param_dim,fc_layers=3,fc_layer_dim=256, lstm_num_layers=1,apply_dropout=False,dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.feature_size=feature_size
        self.hidden_dim = RNN_hidden_dim
        self.num_layers = lstm_num_layers
        self.dropout = dropout
        self.rnn = nn.LSTM(feature_size,RNN_hidden_dim, lstm_num_layers,batch_first=True)

        self.fc_input_layer=nn.Linear(RNN_hidden_dim+param_dim,fc_layer_dim)


        self.fc_sequence=Res_ANN_new(n_layers=fc_layers,layer_dim=fc_layer_dim,apply_dropout=apply_dropout,dropout=dropout)
        self.output=nn.Linear(fc_layer_dim,1)
        


    def forward(self, x,params):
        

        output, (hn, cn) = self.rnn(x)

        fc_input=self.fc_input_layer(torch.cat((hn[-1],params),dim=1))
        fc_output=self.fc_sequence(fc_input)
        output=torch.sigmoid(self.output(fc_output))
 
        return output


class ANN_Classifier_Res(torch.nn.Module):
    def __init__(self, input_dim,param_dim,inter_fc_layers=3,fc_layer_dim=256, apply_dropout=False,dropout=0.2):
        super().__init__()
        self.input_dim=input_dim

        self.inter_fc_layers = inter_fc_layers
        self.dropout = dropout
        # self.input_dropout=nn.Dropout(0.2)
        self.fc_input_layer=nn.Linear(param_dim+input_dim,fc_layer_dim)

        self.fc_sequence=Res_ANN_new(n_layers=inter_fc_layers,layer_dim=fc_layer_dim,apply_dropout=apply_dropout,dropout=dropout)

        self.output=nn.Linear(fc_layer_dim,1)
            


    def forward(self, x,params):
        # x=self.input_dropout(x)
        fc_input=self.fc_input_layer(torch.cat((x,params),dim=1))
        fc_output=self.fc_sequence(torch.relu(fc_input))
        # output=torch.sigmoid(self.output(fc_output))
        output=(self.output(fc_output))
 
        return output


class RNN_ANNclassifier(nn.Module):
    def __init__(self, seq_len, feature_size, param_dim ,RNN_hidden_dim,RNN_num_layers,fc_hidden_dims:list,apply_dropout=True,dropout=0.2):
        super().__init__()
        self.rnn = nn.LSTM(feature_size,RNN_hidden_dim, RNN_num_layers,batch_first=True)
        self.ann_classifier=ANN_Classifier(RNN_hidden_dim,param_dim,fc_hidden_dims,apply_dropout,dropout)


    def forward(self, x,params):
        

        output, (hn, cn) = self.rnn(x)
        ann_input_data=hn[-1]
        ann_output=self.ann_classifier(ann_input_data,params)
        

        return ann_output#,ann_input_data#for summary


class RNN_ResANNclassifier(nn.Module):
    def __init__(self, seq_len, feature_size, param_dim ,RNN_hidden_dim,RNN_num_layers,iter_fc_layers=5,fc_hidden_dims=256,apply_dropout=True,dropout=0.2):
        super().__init__()
        self.rnn = nn.LSTM(feature_size,RNN_hidden_dim, RNN_num_layers,batch_first=True)
        self.ann_classifier=ANN_Classifier_Res(RNN_hidden_dim,param_dim,iter_fc_layers,fc_hidden_dims,apply_dropout,dropout)


    def forward(self, x,params):
        

        output, (hn, cn) = self.rnn(x)
        ann_input_data=hn[-1]
        ann_output=self.ann_classifier(ann_input_data,params)
        

        return ann_output#,ann_input_data#for summary



class RNN_ANNclassifier_withSum(nn.Module):
    def __init__(self, seq_len, feature_size, param_dim ,RNN_hidden_dim,RNN_num_layers,fc_hidden_dims:list,apply_dropout=True,dropout=0.2):
        super().__init__()
        self.rnn = nn.LSTM(feature_size,RNN_hidden_dim, RNN_num_layers,batch_first=True)
        self.ann_classifier=ANN_Classifier(RNN_hidden_dim,param_dim,fc_hidden_dims,apply_dropout,dropout)


    def forward(self, x,params):
        

        output, (hn, cn) = self.rnn(x)
        ann_input_data=hn[-1]
        ann_output=self.ann_classifier(ann_input_data,params)
        

        return ann_output,ann_input_data#for summary