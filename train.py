# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# OpenVPRLab: https://github.com/amaralibey/OpenVPRLab
#
# Licensed under the MIT License. See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import TensorBoardLogger
from src.core.datamodule import DataModule
from src.utils.modules_manager import create_model
from src.utils.config_manager import parse_args

from rich.traceback import install
install() # this is for better traceback formatting

# we mostly use mean and std of ImageNet dataset for normalization
# you can define your own mean and std values and use them
IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

# This is called when the train mode is selected
def train(config):
    seed_everything(config["seed"], workers=True)
    torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True)
    torch.backends.cuda.enable_flash_sdp(True)

    # let's create the VPR DataModule
    datamodule = DataModule(config["datamodule"])

    vpr_model = create_model(config)

    if config["compile"]:
        vpr_model = torch.compile(vpr_model)


    # Let's define the TensorBoardLogger
    # We will save under the logs directory 
    # and use the backbone name as the subdirectory
    # e.g. a BoQ model with ResNet50 backbone will be saved under logs/ResNet50/BoQ
    # this makes it easy to compared different aggregators with the same backbone
    tensorboard_logger = TensorBoardLogger(
        save_dir=f"./logs/{config['backbone']['model']}",
        name=f"{config["aggregator"]["method"]}",
        default_hp_metric=False
    )
    
    # Let's define the checkpointing.
    # We use a callback and give it to the trained
    # The ModelCheckpoint callback saves the best k models based on a validation metric
    # In this example we are using msls-val/R1 as the metric to monitor
    # The checkpoint files will be saved in the logs directory (which we defined in the TensorBoardLogger)
    #TODO: UPDate this to come from config
    checkpoint_cb = ModelCheckpoint(
        monitor=config["datamodule"]["val_set_names"][0] + "/R1",
        filename="epoch({epoch:02d})_step({step:04d})_R1[{eiffel/R1:.4f}]_R5[{eiffel/R5:.4f}]",
        auto_insert_metric_name=False,
        save_weights_only=False,
        save_top_k=3,
        mode="max",
    )
    
    # Let's define the progress bar, model summary and data summary callbacks
    from src.utils.callbacks import CustomRichProgressBar, CustomRRichModelSummary, DatamoduleSummary
    # there are multiple themes you can choose from. They are defined in src.utils.callbacks
    # example: default, cool_modern, vibrant_high_contrast, green_burgundy, magenta
    progress_bar_cb = CustomRichProgressBar(config["display_theme"])    
    model_summary_cb = CustomRRichModelSummary(config["display_theme"])    
    data_summary_cb = DatamoduleSummary(config["display_theme"])
     

    trainer = Trainer(
        accelerator="gpu",
        devices=[0],
        logger=tensorboard_logger,
        num_sanity_val_steps=0, # is -1 to run one pass on all validation sets before training starts
        precision="16-mixed",
        max_epochs=config['trainer']['max_epochs'],
        check_val_every_n_epoch=1,
        callbacks=[
            checkpoint_cb,
            data_summary_cb,    # this will print the data summary
            model_summary_cb,   # this will print the model summary
            progress_bar_cb,    # this will print the progress bar
            ],
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=10,
        fast_dev_run=config["dev"], # dev mode (only runs one train iteration and one valid iteration, no checkpointing and no performance tracking).
        enable_model_summary=False, # we are using our own model summary
    )

    # save the config into logs directory
    # with open(f"{tensorboard_logger.log_dir}/custom_config.yaml", 'w') as file:
    #     yaml.dump(config, file)
    
    trainer.fit(model=vpr_model, datamodule=datamodule)

def evaluate(config):
    print("Evaluation mode selected.")
    # Your evaluation logic here


if __name__ == "__main__":
    config = parse_args()
    
    train(config)
