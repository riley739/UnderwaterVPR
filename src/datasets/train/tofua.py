# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# OpenVPRLab: https://github.com/amaralibey/OpenVPRLab
#
# Licensed under the MIT License. See LICENSE file in the project root.
# ----------------------------------------------------------------------------

"""
GSV-Cities dataset 
====================

This module implements a PyTorch Dataset class for GSV-Cities dataset from the paper:

"GSV-Cities: Toward Appropriate Supervised Visual Place Recognition" 
by Ali-bey et al., published in Neurocomputing, 2022.


Citation:
    @article{ali2022gsv,
        title={{GSV-Cities}: Toward appropriate supervised visual place recognition},
        author={Ali-bey, Amar and Chaib-draa, Brahim and Gigu{\`e}re, Philippe},
        journal={Neurocomputing},
        volume={513},
        pages={194--203},
        year={2022},
        publisher={Elsevier}
    }

URL: https://arxiv.org/abs/2210.10239
"""

import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from src.datasets.train.base_dataset import BaseTrainDataset
import torchvision.transforms as T


# Now we can define the dataset class
class TofuaTrainDataset(BaseTrainDataset):
    def __init__(self,
                 **kwargs
                 ):
        super().__init__(**kwargs)
