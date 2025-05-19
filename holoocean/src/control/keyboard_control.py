import numpy as np
from pynput import keyboard
# import numpy as keyboard
from loguru import logger 

class KeyBoardController():
    def __init__(self):
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.listener.start()

        self.pressed_keys = list()
        self.force = 25
        self.running=True

        # self.forward = [1.0, 0.0, 0.0]
        # self.left = [0.0, 1.0, 0.0]
        # self.backward = [-1.0, 0.0, 0.0]
        # self.right = [0.0, -1.0, 0.0]
        # self.up = [0.0, 0.0, 1.0]
        # self.down = [0.0, 0.0, -1.0]
        # self.stop = [0.0, 0.0, 0.0]
        # self.command = np.zeros(6)

    def on_press(self, key):
        if hasattr(key, 'char'):
            self.pressed_keys.append(key.char)
            self.pressed_keys = list(set(self.pressed_keys))

    def on_release(self, key):
        if hasattr(key, 'char'):
            try:
                self.pressed_keys.remove(key.char)
            except ValueError:
                pass
            
    def exit(self):
        self.listener.stop()
    
    def is_running(self):
        return self.listener.running

    def parse_keys(self):
        command = np.zeros(8)
        if 'i' in self.pressed_keys:
            command[0:4] += self.force
        if 'k' in self.pressed_keys:
            command[0:4] -= self.force
        if 'j' in self.pressed_keys:
            command[[4,7]] += self.force
            command[[5,6]] -= self.force
        if 'l' in self.pressed_keys:
            command[[4,7]] -= self.force
            command[[5,6]] += self.force

        if 'w' in self.pressed_keys:
            command[4:8] += self.force
        if 's' in self.pressed_keys:
            command[4:8] -= self.force
        if 'a' in self.pressed_keys:
            command[[4,6]] += self.force
            command[[5,7]] -= self.force
        if 'd' in self.pressed_keys:
            command[[4,6]] -= self.force
            command[[5,7]] += self.force
        
        if'q' in self.pressed_keys:
            self.exit()

        return command

    def get_command(self, state):
        
        if self.is_running():
            command = self.parse_keys()
            return command
        
        # If not running return stationary 
        logger.warning("Keyboard controller has stopped running, vehicle is unable to be controlled")
        return np.zeros(8)
            
#TODO Convert into different platforms
class Torpedo_Controls():
    def __init__(self):
        MOVE_FORWARD = [1.0, 0.0, 0.0]
        MOVE_BACKWARD = [-1.0, 0.0, 0.0]
        MOVE_LEFT = [0.0, 1.0, 0.0]
        MOVE_RIGHT = [0.0, -1.0, 0.0]
        MOVE_UP = [0.0, 0.0, 1.0]
        MOVE_DOWN = [0.0, 0.0, -1.0]
        STOP = [0.0, 0.0, 0.0]

