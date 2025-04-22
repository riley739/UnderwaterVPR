from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


class SonarWidget(QWidget):
    def __init__(self, config):
        super().__init__()
        self.running = False

        self.setup(config)

    def update_data(self, data):
        if self.running:
            if data.shape != self.z.shape:
                    print(f"Warning: sonar shape mismatch. Got {data.shape}, expected {self.z.shape}")
                    return
            
            self.plot.set_array(data.ravel())
            self.plot.changed()  # Mark for redraw
            self.canvas.draw()

    
    def setup(self, config):
        sensors = config["agents"][0]["sensors"]

        for sensor in sensors:
            if sensor["sensor_type"] == "ImagingSonar":
                configuration = sensor["configuration"]
                break
        else:
            return
        
        # Read configuration    
        self.azi = configuration['Azimuth']
        self.minR = configuration['RangeMin']
        self.maxR = configuration['RangeMax']
        self.binsR = configuration['RangeBins']
        self.binsA = configuration['AzimuthBins']

        # Calculate angle and range grids
        self.theta = np.linspace(-self.azi/2, self.azi/2, self.binsA) * np.pi / 180
        self.r = np.linspace(self.minR, self.maxR, self.binsR)
        self.T, self.R = np.meshgrid(self.theta, self.r)
        self.z = np.zeros_like(self.T)

        # Setup figure and polar axis
        self.figure = Figure(figsize=(8, 5))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, projection='polar')
        self.ax.set_theta_zero_location("N")
        self.ax.set_thetamin(-self.azi / 2)
        self.ax.set_thetamax(self.azi / 2)
        self.ax.grid(False)

        self.plot = self.ax.pcolormesh(self.T, self.R, self.z, cmap='copper', shading='auto', vmin=0, vmax=1)


        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.running = True


