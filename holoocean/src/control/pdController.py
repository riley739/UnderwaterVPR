import numpy as np
from loguru import logger


class pdController():
    def __init__(self, location):
        
        base_x = location[0]
        base_y = location[1]
        height = location[2] 

        grid_length =  25
        grid_width = 3
        number_rows = 20
        locations = []

        self.running = True

        for i in range(0,number_rows,2):
            if i == number_rows//2:
                height += 10
            locations.append([base_x - grid_length,  base_y -  grid_width*i, height ])
            locations.append([base_x - grid_length,  base_y - grid_width*(i+1), height])
            locations.append([base_x,  base_y - grid_width*(i+1),  height])
            locations.append([base_x, base_y -  grid_width*(i+2), height])
    
        self.idx = 0
        self.number_locations = len(locations)
        self.locations = np.array(locations)
        logger.info(f"Going to waypoint {self.idx} / {self.number_locations}")


    def get_command(self, state):

        if "PoseSensor" in state:
            pose = state["PoseSensor"][:3, 3]
            if np.linalg.norm(pose-self.locations[self.idx]) < 0.5:
                self.idx = (self.idx+1) % self.number_locations
                logger.info(f"Going to waypoint {self.idx} / {self.number_locations}")
        
                if self.idx == self.number_locations:
                    self.running = False

        command = self.locations[self.idx]
        return command
