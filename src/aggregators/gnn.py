from typing import List, Optional

import torch
from torch import nn

# Start Simple and then add more features when it works e.g. 

# Get working with places and no extra data -> e.g. just visual data going into the gnn 

# Then add the heading values 

# Then test on pose with the simulator

# Then add with sonar data

# Then find way of extracting cnn data

# hopefully still produces ok data... 

#We can hack this togehter for now but future methods should have reranking capabilities 
# For each image in the batch:
    # So i Think here what we do is figure out way to get it as one global descriptor.
    # Can try an agregator can also try cls token 

    # Then we need to get the topk database images
    # pass those  desc and affinity vectors and pass that through the gnn ... 
    # this then returns the new descriptors 
    # this then goes through loss function as expected 
    # I think this is defs plausible to do 



class GnnLayer(nn.Module):
    def __init__(
        self, descriptor_dim: int, num_heads: int, attn_dropout_p: float = 0.0, hidden_dim: Optional[int] = None
    ) -> None:
        """! Class initializer.

        @param descriptor_dim Dimension of image descriptors.
        @param num_heads Number of attention heads.
        @param attn_dropout_p Attention dropout probability.
        @param hidden_dim Dimension of hidden layer in MLP.
        """
        super().__init__()
        self.attn_norm = nn.LayerNorm(descriptor_dim)
        self.attn = nn.MultiheadAttention(descriptor_dim, num_heads, dropout=attn_dropout_p, batch_first=True)
        self.mlp_norm = nn.LayerNorm(descriptor_dim)
        hidden_dim = hidden_dim or 4 * descriptor_dim
        self.mlp = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, descriptor_dim),
        )

    def forward(self, desc: torch.Tensor) -> torch.Tensor:
        """! Forward pass.

        @param desc Image descriptors (B, N, D).
        @return The updated descriptors (B, N, D).
        """
        
        desc_norm = self.attn_norm(desc)
        attn, _ = self.attn(desc_norm, desc_norm, desc_norm)
        desc = desc + attn
        desc_norm = self.mlp_norm(desc)
        desc = desc + self.mlp(desc_norm)
        return desc


class GNN(nn.Module):
    def __init__(
        self,
        config,
    ) -> None:
        """! Class initializer.

        @param input_dim Dimension of input image descriptors.
        @param proj_dim Dimension of projected image descriptors.
        @param output_dim Dimension of output image descriptors.
        @param K The number of database images to compute new descriptors for.
        @param L The number of database images to use for calculating affinity features.
        @param num_layers Number of self-attention layers.
        @param num_heads Number of self-attention heads in each layer.
        @param aff_dim Dimension of non-visual affinity vectors.
        @param input_dropout_p Input dropout probability.
        @param attn_dropout_p Attention dropout probability.
        """
        super().__init__()

        self.input_dropout = nn.Dropout(config.get("input_dropout_p", 0))

        self.input_proj = nn.Identity() if config["input_dim"] == config["proj_dim"] else nn.Linear(config["input_dim"], config["proj_dim"])

        if config["num_layers"] > 0:

            dim = config['L'] + 1 #+ config["aff_dim"]
            self.affinity_proj = nn.Identity() if dim == config["output_dim"] else nn.Linear(dim, config["output_dim"])

        self.layers = nn.ModuleList([GnnLayer(config["output_dim"], config["num_heads"], config.get("attn_dropout_p",0)) for _ in range(config["num_layers"])])

        self.K = config["K"]
        self.L = config["L"]
        self.aff_dim = config["aff_dim"]

    
    def forward(self, X) -> torch.Tensor:
        """
        x: The global descriptors
        aff: The affinity features 
        Returns:
            f (torch.Tensor): The global descriptor [B, m*l + g]
        """
                
        """! Forward pass.

        @param desc Image descriptors with the query at index 0 ((B, 1+N, D), where N ≥ max(K, L)).
        @param aff Non-visual affinities for the images ((B, N, N) or (B, 1+N, 1+N)).
        @return The updated image descriptors (B, 1+K, D').
        """

        desc, aff = X 
        # x is the global descriptors out of the previous model 
        # x = [images_per_place * batch_size, 3, w,h]
        
        # Have to do faiss with these descriptors and that of the database...
        N = desc.shape[1] - 1
        desc = self.input_dropout(desc)
        desc = self.input_proj(desc)
        desc = nn.functional.normalize(desc, dim=2)

        # If no layers return the projected descriptors, to allow for training the input projection separately
        if len(self.layers) == 0:
            return desc[:, : self.K + 1]

        # Compute visual affinity features
        desc = desc[:, : self.K + 1] @ desc[:, : self.L + 1].transpose(1, 2)  # (B, 1+K, 1+L)

        # Append non-visual affinity vectors
        if self.aff_dim:

            tensors = [desc]

            for a in aff:
                if a.shape[1] == N:
                    aff_query = torch.zeros((a.shape[0], 1, self.L), device=a.device)  # (B, 1, L)
                    tensors.append(torch.cat((aff_query, a[:, : self.K, : self.L]), dim=1))  # (B, 1+K, L)
                else:
                    tensors.append(a[:, : self.K + 1, : self.L + 1])  # (B, 1+K, 1+L)

            desc = torch.cat(tensors, dim=2)  # (B, 1+K, )

        desc = self.affinity_proj(desc)  # (B, 1+K, D')

        # Refine descriptors in GNN layers
        for layer in self.layers:
            desc = layer(desc)

        desc = nn.functional.normalize(desc, dim=2)

        return desc
