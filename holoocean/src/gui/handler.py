import lcm
from src.lcm.messages.PoseSensor import PoseSensor
from src.lcm.messages.RGBCamera import RGBCamera
from src.lcm.messages.ImagingSonar import ImagingSonar
from src.lcm.messages.LoopClosure import LoopClosure
import numpy as np

class LCMHandler:
    def __init__(self, config, comm):
        self.comm = comm

        #TODO Have these come from config
        try:
            self.lc = lcm.LCM(config["lcm_provider"])
        except:
            print("Shutting Down no lcm messages published")
            exit()

        for sensor in config["agents"][0]["sensors"]:
            if "pose" in sensor.get("lcm_channel", "").lower():
                self.lc.subscribe(sensor["lcm_channel"], self.handle_pose)
            elif "rgb" in sensor.get("lcm_channel", "").lower():
                self.lc.subscribe(sensor["lcm_channel"], self.handle_image)
            elif "imaging" in sensor.get("lcm_channel", "").lower():
                self.lc.subscribe(sensor["lcm_channel"], self.handle_imaging_sonar)
        self.lc.subscribe("shutdown", self.handle_shutdown)
        self.lc.subscribe("loopclosure", self.hand_lc)
            
    def handle_pose(self, channel, data):
        msg = PoseSensor.decode(data).matrix
        self.comm.pose_signal.emit(msg[0][-1], msg[1][-1], msg[2][-1])

    def handle_image(self, channel, data):
        msg = RGBCamera.decode(data)
        img = np.array(msg.image, dtype=np.uint8).reshape((msg.height, msg.width, msg.channels))
        self.comm.image_signal.emit(img)
    
    def handle_imaging_sonar(self, channel, data):
        msg = ImagingSonar.decode(data).image
        self.comm.sonar_signal.emit(msg)

    def handle_shutdown(self, channel, data):
        self.comm.shutdown_signal.emit()

    def hand_lc(self, channel, data):
        msg = LoopClosure.decode(data)
        self.comm.loopclosure_signal.emit(msg)
        
    def handle_once(self):
        # Non-blocking version
        self.lc.handle_timeout(1)  # 1 ms timeout
