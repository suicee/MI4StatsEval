import torch
import torch.nn as nn
import torch.nn.functional as F



class ANN(torch.nn.Module,):

    def __init__(self,input_dim,hidden_dims:list,output_dim=1,apply_dropout=True,dropout=0.2) -> None:
        super().__init__()


        self.input=nn.Linear(input_dim,hidden_dims[0])
        
        fc_modules=[]
        
        for i in range(1,len(hidden_dims)):
            fc_modules.append(nn.ReLU())

            if apply_dropout:
                fc_modules.append(nn.Dropout(dropout))
            fc_modules.append(nn.Linear(hidden_dims[i-1],hidden_dims[i]))
            
        fc_modules.append(nn.ReLU())
        self.hidden_sequence=nn.Sequential(*fc_modules)

        self.output=nn.Linear(hidden_dims[-1],output_dim)




    def forward(self,x):

        inp=self.input(x)

        lat=self.hidden_sequence(inp)

        out=self.output(lat)

        return out


class ANN_Classifier(torch.nn.Module,):

    def __init__(self,data_dim,para_dim,hidden_dims:list,apply_dropout=True,dropout=0.2) -> None:
        super().__init__()

        self.net=ANN(data_dim+para_dim,hidden_dims,1,apply_dropout,dropout)


    def forward(self,x,params):

        out=self.net(torch.cat((x,params),dim=-1))
        # out=torch.sigmoid(out)

        return out
