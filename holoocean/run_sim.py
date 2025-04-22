import argparse
import json
from loguru import logger
import lcm
from src.logger import setup_logger
from src.simulation import simulation
from src.utils import load_config
from src.lcm.messages import ShutDown

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read a config file as an argument.")
    parser.add_argument("--config_file", type=str, default = "configs/default.json", help="Path to the configuration file")

    #TODO: Update this to be like the other methods in underwater vpr 
    args = parser.parse_args()
    config = load_config(args.config_file)

    config["log_dir"] = setup_logger(config["log_dir"])

    try:
        simulation(config)
    finally:
        lc = lcm.LCM(config["scenario"]["lcm_provider"])
        msg = ShutDown()
        lc.publish("shutdown", msg.encode())
        logger.info("Simulation Stopping, Shutdown message published")
