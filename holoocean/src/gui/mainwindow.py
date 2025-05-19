from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSizePolicy
from .camerafeed import ImageDisplayWidget
from .plotfeed import LivePlotWidget
from .sonarfeed import SonarWidget

class MainWindow(QMainWindow):
    def __init__(self, config, communicator):
        super().__init__()
        self.setWindowTitle("Simulation Output")

        # Widgets
        self.plot_widget = LivePlotWidget(config)
        self.image_widget = ImageDisplayWidget("RGB Camera")
        self.sonar_widget = SonarWidget(config)


        # Main layout
        central = QWidget()
        main_layout = QHBoxLayout()

        # Left side - Plot
        main_layout.addWidget(self.plot_widget, stretch=1)

        # Right side - Vertical stack of image + sonar
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.image_widget, stretch=1)
        right_layout.addWidget(self.sonar_widget, stretch=1)

        # Container for right-side widgets
        right_container = QWidget()
        right_container.setLayout(right_layout)

        main_layout.addWidget(right_container, stretch=1)

        central.setLayout(main_layout)
        self.setCentralWidget(central)

        # Signal connections
        communicator.pose_signal.connect(self.plot_widget.add_point)
        communicator.image_signal.connect(self.image_widget.update_image)
        communicator.sonar_signal.connect(self.sonar_widget.update_data)
        communicator.shutdown_signal.connect(self.close)
        communicator.loopclosure_signal.connect(self.plot_widget.update_points)
