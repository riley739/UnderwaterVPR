from src.datasets.val.base_dataset import BaseValDataset

class PittsburghValDataset(BaseValDataset):
    def __init__(
        self,
        **kwargs
    ):
       super().__init__(**kwargs)
