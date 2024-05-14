import sys
import random
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib
matplotlib.use('Qt5Agg')

from PyQt5 import QtCore, QtWidgets, QtGui

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

if QtCore.qVersion() >= "5.":
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)

class MplCanvas(QtWidgets.QWidget):
    def __init__(self, parent=None, width=5, height=4, dpi=100, toolbar=True):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        self.setLayout(layout)
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.canvas = FigureCanvas(self.fig)
        self.axes = self.fig.gca(projection='3d')
        self.axes.set_xlim3d(-0.5, 0.5)
        self.axes.set_ylim3d(-0.5, 0.5)
        self.axes.set_zlim3d(-0.5, 0.5)
        self.fig.tight_layout(pad=-1)
        self._plot_ref = None
        self._border_ref = None
        self.axes.set_facecolor((0, 0, 0))

        layout.addWidget(self.canvas)
        if toolbar:
            if parent:
                parent.addToolBar(NavigationToolbar(self.canvas, self))
            else:
                raise ValueError('set parent to QMainWindow instance in order to use toolbar')
        self.axes.axis('off')
    
    def update_points(self, new_points):
        # Note: we no longer need to clear the axis.
        new_points = new_points.copy()
        new_points[:, 2] = new_points[:, 2] / 0.75
        if self._plot_ref is None:
            # First time we have no plot reference, so do a normal plot.
            # .plot returns a list of line <reference>s, as we're
            # only getting one we can take the first element.
            _plot_refs = self.axes.plot(new_points[:, 0],\
                                               new_points[:, 1],\
                                               new_points[:, 2], 
                                               marker='.', 
                                               linestyle='none',
                                               color=(1, 1, 1),
                                               markersize=1)
            self._plot_ref = _plot_refs[0]
        else:
            self._plot_ref.set_data_3d(new_points[:, 0],\
                                       new_points[:, 1],\
                                       new_points[:, 2], )

        self.canvas.draw()
        


def generate_3d_points(count):
    return np.random.random((count, 3)) * 2 - 1

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.plot3D = MplCanvas(self, width=20, height=20, dpi=130, toolbar=False)
        self.test_text = QtWidgets.QTextEdit("Hello world")

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self.test_text)
        splitter.addWidget(self.plot3D)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([125, 150])

        self.setCentralWidget(splitter)

        self.max_points = 30000
        self.update_per_run = 300

        self.data_points = generate_3d_points(self.max_points)

        # We need to store a reference to the plotted line 
        # somewhere, so we can apply the new data to it.
        self.update_plot()

        self.show()

        # Setup a timer to trigger the redraw by calling update_plot.
        self.timer = QtCore.QTimer()
        self.timer.setInterval(1)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

    def update_plot(self):
        # Drop off the first y element, append a new one.
        self.data_points = np.concatenate((self.data_points[self.update_per_run:],\
                                            generate_3d_points(self.update_per_run)))
        self.plot3D.update_points(self.data_points)

        



from qasync import QEventLoop, QThreadExecutor
import asyncio
if __name__ == '__main__':
    # for port, desc, hwid in sorted(ports):
            # print("{}: {} [{}]".format(port, desc, hwid))
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    ex = MainWindow()
    ex.show()

    with loop:
        sys.exit(loop.run_forever())