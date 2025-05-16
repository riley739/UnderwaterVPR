from enum import Enum
from loguru import logger
from .keyboard_control import KeyBoardController
from .pdController import pdController

class ControlMode(Enum):
    KEYBOARD = 0
    PD = 1

class Controller():
    def __init__(self, config):
        control_mode = config["control_scheme"]
        self.running = True
        if control_mode == ControlMode.KEYBOARD.value:
            logger.info(f"Operating in control mode {ControlMode(control_mode).name}")
            self.controller = KeyBoardController()
        elif control_mode == ControlMode.PD.value:
            logger.info(f"Operating in control mode {ControlMode(control_mode).name}")
            self.controller = pdController(config.get("location", [0,0,0]))
        else:
            logger.warning(f"Invalid Control value {control_mode} was given, vehicle is uncontrollable")
    
    def get_command(self, state):
        #TODO: this is like one frame later but doesn't really matter right?
        if self.controller.running:
            return self.controller.get_command(state)
        else:
            return []
    

    def is_running(self):
        return self.controller.running