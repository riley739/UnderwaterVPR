import cv2
import os
import glob
from loguru import logger
import threading 

#TODO: Only works with one sonar and one sonar type expand later..
class SonarReader():
    def __init__(self, config):

        self.sonar_running = False

        self._setup(config)

    def save_data(self, state):
        if self.sonar_running:

            logger.log("SONAR", f"Sonar data: {state["ImagingSonar"]}\nPose: {state['PoseSensor']}")


    def _setup(self, config):

        sensors = config["scenario"]["agents"][0]["sensors"]

        for sensor in sensors:
            if sensor["sensor_type"] == "ImagingSonar":
                self.sonar_running = True

                logger.log("SONAR", f"Configuration values: {sensor["configuration"]}")

                logger.info("Sonar found")
                return
            
        logger.info("Sonar not found")



