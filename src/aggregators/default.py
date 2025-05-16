import torch.nn as nn

#Default Model - Hack to allow for the training of the backbone without the aggregator effecting it 
class Default(nn.Module):
     
    def __init__(
        self,
        config
    ):
        super().__init__()
        
      

    def forward(self, x):

        return x
