from src.datasets.val.base_dataset import BaseValDataset

class TofuaValDataset(BaseValDataset):
    def __init__(
        self,
        **kwargs
    ):
       super().__init__(**kwargs)
