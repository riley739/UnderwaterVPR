"""
    BoQ: A Place is Worth a Bag of learnable Queries (CVPR 2024)
    
    Paper: https://arxiv.org/abs/2405.07364
    GitHub repo: https://github.com/amaralibey/Bag-of-Queries
    
    Reference:
    @InProceedings{Ali-bey_2024_CVPR,
        author    = {Ali-bey, Amar and Chaib-draa, Brahim and Gigu\`ere, Philippe},
        title     = {{BoQ}: A Place is Worth a Bag of Learnable Queries},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month     = {June},
        year      = {2024},
        pages     = {17794-17803}
    }
    
"""


import torch

DINOV2 = "Dinov2"
RESNET50 = 'ResNet50'

MODEL_URLS = {
    f"{RESNET50}_16384": "https://github.com/amaralibey/Bag-of-Queries/releases/download/v1.0/resnet50_16384.pth",
    f"{DINOV2}_12288": "https://github.com/amaralibey/Bag-of-Queries/releases/download/v1.0/dinov2_12288.pth",
}

class BoQBlock(torch.nn.Module):
    """
    BoQ Block

    Args:
        in_dim (int): input dimension
        config["num_queries"] (int): number of queries to learn
        nheads (int): number of heads in the multihead attention. Defaults to 8.
    """
    def __init__(self, in_dim, num_queries, nheads=8):
        super(BoQBlock, self).__init__()
        
        self.encoder = torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=nheads, dim_feedforward=4*in_dim, batch_first=True, dropout=0.)
        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_dim))
        
        # the following two lines are used to add context between the learned queries during training. 
        # you can cache their output in eval.
        self.self_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_q = torch.nn.LayerNorm(in_dim)
        ################
        
        self.cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_out = torch.nn.LayerNorm(in_dim)
        

    def forward(self, x):
        B = x.size(0)
        x = self.encoder(x)
        
        q = self.queries.repeat(B, 1, 1)
        
        q = q + self.self_attn(q, q, q)[0]
        q = self.norm_q(q)
        
        out, attn = self.cross_attn(q, x, x)
        out = self.norm_out(out)
        return x, out, attn.detach()


class BoQ(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        # reduce input dimension using 3x3 conv
        self.proj_c = torch.nn.Conv2d(config["in_channels"], config["proj_channels"], kernel_size=3, padding=1)
        
        # normalize the input to the BoQ blocks
        self.norm_input = torch.nn.LayerNorm(config["proj_channels"])
        
        # now the BoQ blocks input dimension is config["proj_channels"]
        boq_in_dim = config["proj_channels"]
        
        # create the BoQ blocks (each head of the self-attention has a dimension of 64)
        self.boqs = torch.nn.ModuleList([
            BoQBlock(boq_in_dim, config["num_queries"], nheads=boq_in_dim//64) for _ in range(config["num_layers"])])
        
        # the outputs of all BoQ blocks are concatenated and projected to config["row_dim"]
        self.fc = torch.nn.Linear(config["num_layers"]*config["num_queries"], config["row_dim"])
        
    def forward(self, x):
        x = self.proj_c(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.norm_input(x)
        
        outs = []
        attns = []
        for i in range(len(self.boqs)):
            x, out, attn = self.boqs[i](x)
            outs.append(out)
            attns.append(attn)

        out = torch.cat(outs, dim=1)
        out = self.fc(out.permute(0, 2, 1))
        out = out.flatten(1)
        out = torch.nn.functional.normalize(out, p=2, dim=-1)
        return out, attns
