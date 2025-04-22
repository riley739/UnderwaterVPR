from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import time
# ──────────────────────────────────────────────
# 📊 Plot Widget
# ──────────────────────────────────────────────
class LivePlotWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax_xy = self.figure.add_subplot(211)
        self.ax_y_time = self.figure.add_subplot(212)
        

        self.line_xy, = self.ax_xy.plot([], [], lw=2)
        self.line_y_time, = self.ax_y_time.plot([], [], lw=2)

        self.x_data = []
        self.y_data = []
        self.z_data = []
        self.time_data = []
        self.start_time = None

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.ax_xy.set_title("XY Trajectory")
        self.ax_xy.set_xlabel("X")
        self.ax_xy.set_ylabel("Y")
        self.ax_xy.grid()

        self.ax_y_time.set_title("Y over Time")
        self.ax_y_time.set_xlabel("Time (s)")
        self.ax_y_time.set_ylabel("Y")
        self.ax_y_time.grid()
        self.max_threshold = 2000


    def add_point(self, x, y, z):
        if self.start_time == None:
            self.start_time = time.time()
            t = 0
        else:
            t = time.time() - self.start_time
        self.x_data.append(x)
        self.y_data.append(y)
        self.z_data.append(z)
        self.time_data.append(t)

        if len(self.x_data) > self.max_threshold:
            self.x_data = self.x_data[-self.max_threshold:]
            self.y_data = self.y_data[-self.max_threshold:]
            self.z_data = self.z_data[-self.max_threshold:]
            self.time_data = self.time_data[-self.max_threshold:]

        
        self.update_plot()

    def update_plot(self):
        # x is forward and back, y is left and right
        self.line_xy.set_data(self.y_data, self.x_data)
        self.line_y_time.set_data(self.time_data, self.z_data)

        if self.x_data:
            self.ax_xy.set_xlim(min(self.y_data) - 1, max(self.y_data) + 1)
            self.ax_xy.set_ylim(min(self.x_data) - 1, max(self.x_data) + 1)
            self.ax_xy.invert_xaxis()

        if self.z_data:
            self.ax_y_time.set_xlim(min(self.time_data), max(self.time_data) + 1) 
            self.ax_y_time.set_ylim(min(self.z_data) - 10, max(self.z_data) + 10)

        self.canvas.draw()
