import sys
import argparse
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import  QTimer
from src.gui.communicator import Communicator 
from src.gui.mainwindow import MainWindow
from src.gui.handler import LCMHandler
from src.utils import load_config

import os
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"


# ──────────────────────────────────────────────
# 🚀 App Launcher
# ──────────────────────────────────────────────
def run_gui(config):
    app = QApplication(sys.argv)

    # Create communicator instance (shared with GUI and data source)
    comm = Communicator()

    # Start GUI
    window = MainWindow(config["scenario"], comm)
    window.resize(1600, 900)

    #Move onto third screen
    screens = QGuiApplication.screens()
    screen = screens[-1]  # Last monitor
    geometry = screen.geometry()
    window.move(geometry.left() + geometry.width()//2 - window.width()//2, geometry.top() + geometry.height()//2 - window.height()//2)
    window.show()


    lcm_handler = LCMHandler(config["scenario"], comm)
    
    timer = QTimer()
    timer.timeout.connect(lcm_handler.handle_once)
    timer.start(10)

    sys.exit(app.exec_())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read a config file as an argument.")
    parser.add_argument("--config_file", type=str, default = "configs/default.json", help="Path to the configuration file")
    parser.add_argument("--database", type=str, default = None, help="Path to the database file")

    #TODO: Update this to be like the other methods in underwater vpr 
    args = parser.parse_args()
    config = load_config(args.config_file)
     
    if args.database:
        config["scenario"]["database"] = args.database

    print(f"Using config file {args.config_file}")

    run_gui(config)
