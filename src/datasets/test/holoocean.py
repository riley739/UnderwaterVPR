from src.datasets.test.base_dataset import BaseTestDataset

class HoloOceanTestDataset(BaseTestDataset):
    def __init__(
        self,
        **kwargs
    ):
       super().__init__(**kwargs)
