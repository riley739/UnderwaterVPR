import torch.nn as nn

#Default Model - Hack to allow for the training of the Aggregator without backbone effecting it
class Default(nn.Module):
     
    def __init__(
        self,
        config
    ):
        super().__init__()
        self.out_channels = 0
      

    def forward(self, x):

        return x
