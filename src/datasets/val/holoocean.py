from src.datasets.val.base_dataset import BaseValDataset

class HoloOceanValDataset(BaseValDataset):
    def __init__(
        self,
        **kwargs
    ):
       super().__init__(**kwargs)
