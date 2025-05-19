from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd 
import time
# ──────────────────────────────────────────────
# 📊 Plot Widget
# ──────────────────────────────────────────────
class LivePlotWidget(QWidget):
    def __init__(self, config):
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

        self.update_loopclosure  = False
        self.correct_star_plot, = self.ax_xy.plot([], [], linestyle='None', marker='*', color='green', markersize=10, zorder=3)
        self.incorrect_star_plot, = self.ax_xy.plot([], [], linestyle='None', marker='*', color='red', markersize=10, zorder=3)
        self.query_star_plot, = self.ax_xy.plot([], [], linestyle='None', marker='*', color='blue', markersize=10, zorder=3)
        

        
        self.reference = False
        if config.get("database"):
            df = pd.read_csv(config["database"])
            self.reference = True
            
            self.database_x = list(df['x'])
            self.database_min_x = min(self.database_x)
            self.database_max_x = max(self.database_x)

            self.database_y = list(df["y"])
            
            self.database_min_y = min(self.database_y)
            self.database_max_y = max(self.database_y)
            #TODO: Make sure its ok to swap them think its just cause of holoocean frame of reference
            self.ax_xy.plot(self.database_y, self.database_x, linestyle=':', marker='o', color='gray', label='Reference')

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

    def update_points(self, msg): 
        self.pose1 = msg.pose1
        self.pose2 = msg.pose2 
        self.pose3 = msg.pose3 

        self.query = msg.query_pose

        self.update_loopclosure = True



    def update_plot(self):
        # x is forward and back, y is left and right
        self.line_xy.set_data(self.y_data, self.x_data)
        self.line_y_time.set_data(self.time_data, self.z_data)

        if self.x_data:
            #Follow trajectory or database whichever one is larger
            if self.reference:
                self.ax_xy.set_xlim(min(self.database_min_y, min(self.y_data)) - 1, max(self.database_max_y, max(self.y_data)) + 1)
                self.ax_xy.set_ylim(min(self.database_min_x, min(self.x_data)) - 1, max(self.database_max_x, max(self.x_data)) + 1)
            else:
                self.ax_xy.set_xlim(min(self.y_data) - 1, max(self.y_data) + 1)
                self.ax_xy.set_ylim(min(self.x_data) - 1, max(self.x_data) + 1)
            self.ax_xy.invert_xaxis()

        if self.z_data:
            self.ax_y_time.set_xlim(min(self.time_data), max(self.time_data) + 1) 
            self.ax_y_time.set_ylim(min(self.z_data) - 10, max(self.z_data) + 10)
        
        if self.update_loopclosure: 
            x_correct = []
            y_correct = []
            x_wrong = []
            y_wrong = [] 

            x_correct.append(self.pose1.position[0]) if self.pose1.is_correct else x_wrong.append(self.pose1.position[0])
            y_correct.append(self.pose1.position[1]) if self.pose1.is_correct else y_wrong.append(self.pose1.position[1])
            
            x_correct.append(self.pose2.position[0]) if self.pose2.is_correct else x_wrong.append(self.pose2.position[0])
            y_correct.append(self.pose2.position[1]) if self.pose2.is_correct else y_wrong.append(self.pose2.position[1])
            
            x_correct.append(self.pose3.position[0]) if self.pose3.is_correct else x_wrong.append(self.pose3.position[0])
            y_correct.append(self.pose3.position[1]) if self.pose3.is_correct else y_wrong.append(self.pose3.position[1])
            
            self.correct_star_plot.set_data(y_correct, x_correct)   # note: (y, x) if plotting left/right vs forward/back
            self.incorrect_star_plot.set_data(y_wrong, x_wrong)

            self.query_star_plot.set_data([self.query.position[1]], [self.query.position[0]])

            self.update_loopclosure  = False

        self.canvas.draw()
