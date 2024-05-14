import sys, datetime
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

import os, traceback

from points_parser import PointsManager, parse_points

class MplCanvas(QtWidgets.QWidget):
    def __init__(self, parent=None, width=5, height=4, dpi=100): #, toolbar=True)
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
        

        layout.addWidget(self.canvas)
        # if toolbar:
        #     if parent:
        #         parent.addToolBar(NavigationToolbar(self.canvas, self))
        #     else:
        #         raise ValueError('set parent to QMainWindow instance in order to use toolbar')
        # self.axes.set_facecolor((0, 0, 0))
        # self.axes.axis('off')
        # self.axes.grid(False)

        # 
        self._points_manager = None
        self._points_manager_changed = True
        self._lines_refs = None

    @property
    def points_manager(self):
        return self._points_manager
    
    @points_manager.setter
    def points_manager(self, value):
        if value is None:
            self._points_manager = None
        
        assert isinstance(value, PointsManager), 'Invalid type'
        self._points_manager = value
        self._points_manager_changed = True
        self._groups_updated = True

    def change_points_size(self):
        self._groups_updated = True

    def update_points(self):
        if not self._points_manager:
            return # Пока ещё не было загружено ни каких точек
        if self._points_manager_changed or self._groups_updated: # Первый запуск осле смены points_manager, надо всё переинициализировать
            self._points_manager_changed = False
            self._groups_updated = False
            self._lines_refs = []
            self.axes.clear()
            # self.axes.axis('off')
            # self.axes.grid(False)

            x_max = None
            x_min = None
            y_max = None
            y_min = None
            z_max = None
            z_min = None
            for group in self._points_manager:
                xs = group['xs']
                ys = group['ys']
                zs = group['zs'] / 0.75
                
                x_min_t = np.min(xs)
                x_max_t = np.max(xs)
                y_min_t = np.min(ys)
                y_max_t = np.max(ys)
                z_min_t = np.min(zs)
                z_max_t = np.max(zs)
                if x_max is None:
                    x_min = x_min_t
                    x_max = x_max_t
                    
                    y_min = y_max_t
                    y_max = y_max_t
                    
                    z_min = z_min_t
                    z_max = z_max_t
                else:
                    x_min = min(x_min, x_min_t)
                    x_max = max(x_max, x_max_t)
                    
                    y_min = min(y_min, y_min_t)
                    y_max = max(y_max, y_max_t)
                    
                    z_min = min(z_min, z_min_t)
                    z_max = max(z_max, z_max_t)

                size = group['size']
                color = group['color']
                _plot_refs = self.axes.plot(xs,ys, zs, 
                                               marker='.', 
                                               linestyle='none',
                                               color=color,
                                               markersize=size)
                self._lines_refs.append(_plot_refs[0])


                positions, velocity, weights = self._points_manager.simulation.data
                center_of_mass = np.sum(positions  * weights.reshape(-1, 1), axis=0) / np.sum(weights)
                distances = np.sqrt(np.sum(np.power((positions - center_of_mass), 2), axis=1)) + 0.1
                std_distance_from_center = np.std([0, *distances])
                x_min = center_of_mass[0] - 3 * std_distance_from_center
                x_max = center_of_mass[0] + 3 * std_distance_from_center
                y_min = center_of_mass[1] - 3 * std_distance_from_center
                y_max = center_of_mass[1] + 3 * std_distance_from_center
                z_min = center_of_mass[2] - 3 * std_distance_from_center
                z_max = center_of_mass[2] + 3 * std_distance_from_center

                print('center_of_mass', center_of_mass)
                print('distances', distances)
                print('std_distance_from_center', std_distance_from_center)

                self.axes.set_xlim3d(x_min, x_max)
                self.axes.set_ylim3d(y_min, y_max)
                self.axes.set_zlim3d(z_min, z_max)
        else:
            for ind, group in enumerate(self._points_manager):
                xs = group['xs']
                ys = group['ys']
                zs = group['zs'] / 0.75
                self._lines_refs[ind].set_data_3d(
                    xs, ys, zs
                )

        self.canvas.draw()

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.plot3D = MplCanvas(self, width=20, height=20, dpi=130)
        self.test_text = QtWidgets.QTextEdit("Hello world")

        left_col_base = QtWidgets.QWidget(self)
        left_col = QtWidgets.QVBoxLayout(left_col_base)
        left_col.setAlignment(QtCore.Qt.AlignTop)

        labelInfoPath = QtWidgets.QLabel("Путь к данным симуляции")

        # add to left col
        left_col.addWidget(labelInfoPath)

        self.dataPathLineEdit = QtWidgets.QLineEdit()
        self.selectFileBtn = QtWidgets.QPushButton('...')
        self.selectFileBtn.clicked.connect(self.openSelectFileWithData)

        select_file_layout = QtWidgets.QHBoxLayout()
        select_file_layout.addWidget(self.dataPathLineEdit)
        select_file_layout.addWidget(self.selectFileBtn)

        # add to left col
        left_col.addLayout(select_file_layout)

        self.loadDataFromFile = QtWidgets.QPushButton("Загрузить данные")
        self.loadDataFromFile.clicked.connect(self.loadPointsFromFile)
        
        # add to left col
        left_col.addWidget(self.loadDataFromFile)

        # Размер шага
        time_step_layout = QtWidgets.QHBoxLayout()
        
        time_step_label = QtWidgets.QLabel('dt = ')
        self.time_step_select = QtWidgets.QDoubleSpinBox ()
        self.time_step_select.setMaximum(3*10**16)
        self.time_step_select.setMinimum(0.000001)
        self.time_step_select.setSingleStep(0.000001)
        self.time_step_select.setDecimals(6)
        self.time_step_select.setValue(0.001)

        time_step_layout.addWidget(time_step_label)
        time_step_layout.addWidget(self.time_step_select)

        # add to left col
        left_col.addLayout(time_step_layout)

        # Размеры объектов
        left_col.addWidget(QtWidgets.QLabel('Размеры точек'))
        # min
        min_size_layout = QtWidgets.QHBoxLayout()
        min_size_layout.addWidget(QtWidgets.QLabel('Min ='))
        self.min_size_select = QtWidgets.QDoubleSpinBox ()
        self.min_size_select.setRange(0.1, 50)
        self.min_size_select.setSingleStep(0.1)
        self.min_size_select.setValue(0.9)
        self.min_size_select.setEnabled(False)
        self.min_size_select.valueChanged.connect(self.min_point_size_changed)
        min_size_layout.addWidget(self.min_size_select)
        
        # add to left col
        left_col.addLayout(min_size_layout)

        # max
        max_size_layout = QtWidgets.QHBoxLayout()
        max_size_layout.addWidget(QtWidgets.QLabel('Max ='))
        self.max_size_select = QtWidgets.QDoubleSpinBox ()
        self.max_size_select.setRange(0.1, 50)
        self.max_size_select.setSingleStep(0.1)
        self.max_size_select.setValue(4)
        self.max_size_select.setEnabled(False)
        self.max_size_select.valueChanged.connect(self.max_point_size_changed)
        max_size_layout.addWidget(self.max_size_select)

        # add to left col
        left_col.addLayout(max_size_layout)


        # Обновить размеры точек
        update_points_sizes = QtWidgets.QPushButton("Обновить размеры")
        def update_points():
            self.points_manager.set_min_max_points_sizes(self.min_size_select.value(), self.max_size_select.value())

            # if self.min_size_select.value() != self.points_manager.min_point_size:
            #     self.min_size_select.setValue(self.points_manager.min_point_size)
            # if self.max_size_select.value() != self.points_manager.max_point_size:
            #     self.max_size_select.setValue(self.points_manager.max_point_size)
            self.plot3D.change_points_size()
            self.update_plot()
        update_points_sizes.clicked.connect(update_points)

        # add to left col
        left_col.addWidget(update_points_sizes)

        # Флаг использовать CUDA или нет
        self.cuda_use_flag = QtWidgets.QCheckBox('Использовать CUDA')
        
        # add to left col
        left_col.addWidget(self.cuda_use_flag)

        # decimation
        decimation_layout = QtWidgets.QHBoxLayout()
        decimation_layout.addWidget(QtWidgets.QLabel("Децимация"))
        self.decimation_selection = QtWidgets.QSpinBox()
        self.decimation_selection.setMaximum(20000)
        self.decimation_selection.setMinimum(1)
        self.decimation_selection.setSingleStep(1)
        self.decimation_selection.setValue(100)
        decimation_layout.addWidget(self.decimation_selection)

        # add to left col
        left_col.addLayout(decimation_layout)

        # Запуск / Пауза
        self.start_simulation_btn = QtWidgets.QPushButton('Начать')
        self.start_simulation_btn.setCheckable(True)
        self.start_simulation_btn.clicked.connect(self.clicked_simulation_btn)

        #  fps
        fps_label = QtWidgets.QLabel("FPS:")
        self.fps_show = QtWidgets.QLabel("0")
        fps_layout = QtWidgets.QHBoxLayout()
        fps_layout.addWidget(fps_label)
        fps_layout.addWidget(self.fps_show)

        # add to left col
        left_col.addLayout(fps_layout)

        # Флаг для определения того, когда нужно проводить симуляцию
        self.simulation_running = False

        # add to left col
        left_col.addWidget(self.start_simulation_btn)


        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(left_col_base)
        splitter.addWidget(self.plot3D)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([250, 150])

        self.setCentralWidget(splitter)

        self.update_plot()

        self.show()

    def clicked_simulation_btn(self, el):
        if self.start_simulation_btn.isChecked():
            asyncio.create_task(self.run_simulation())
        else:
            self.stop_simulation()

    # Стартует симуляцию и подготавливает к работе все контролы (отключает те, что запрещено использовать)
    async def run_simulation(self):
        if not self.points_manager:
            print('Не найден self.points_manager')
            return
        
        # disable all
        self.cuda_use_flag.setEnabled(False)
        self.dataPathLineEdit.setEnabled(False)
        self.selectFileBtn.setEnabled(False)
        self.loadDataFromFile.setEnabled(False)
        self.decimation_selection.setEnabled(False)
        self.fps_show.setText("0")

        # изменить надпись на кнопке запуска так, чтоб было Продолжить/Пауза
        self.start_simulation_btn.setText('Пауза')
    
        self.simulation_running = True
        use_cuda = self.cuda_use_flag.isChecked()
        decimation = self.decimation_selection.value()
        time_step = self.time_step_select.value()

        cuda_val = "CUDA" if use_cuda else 'C'
        try:
            ind = 0
            time_prev = datetime.datetime.now()
            while self.simulation_running:
                ind += 1
                old_data = self.points_manager.simulation.data
                await self.points_manager.update(time_step, cuda_val)
                new_data = self.points_manager.simulation.data
                if ind % decimation == 0:
                    self.plot3D.update_points()
                    time_cur = datetime.datetime.now()
                    delta = (time_cur - time_prev).total_seconds()
                    time_prev = time_cur
                    delta = (decimation / delta) * 100 // 100
                    self.fps_show.setText(str(delta))
        except Exception as e:
            self.simulation_running = False
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText('Ошибка при выполнении симуляции')
            msg.setDetailedText('Error: ' + str(e) + '\n' + 'Traceback: ' + traceback.format_exc())
            msg.exec_()
        finally:
            self.cuda_use_flag.setEnabled(True)
            self.dataPathLineEdit.setEnabled(True)
            self.selectFileBtn.setEnabled(True)
            self.loadDataFromFile.setEnabled(True)
            self.decimation_selection.setEnabled(True)
            self.start_simulation_btn.setText('Продолжить')

    def stop_simulation(self):
        self.simulation_running = False

    def min_point_size_changed(self):
        val = self.min_size_select.value()
        if self.max_size_select.value() < val:
            self.min_size_select.setValue(self.max_size_select.value())

    def max_point_size_changed(self):
        val = self.max_size_select.value()
        if self.min_size_select.value() > val:
            self.max_size_select.setValue(self.min_size_select.value())

    def openSelectFileWithData(self):
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "QFileDialog.getSaveFileName()","","Text Files (*.txt);; CSV Files (*.csv)", options=options)
        if fileName:
            self.dataPathLineEdit.setText(fileName)
    
    def loadPointsFromFile(self):
        fileName = self.dataPathLineEdit.text()
        if not os.path.exists(fileName):
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText(f'Указанный файл \"{fileName}\" не найден')
            msg.exec_()
        try:
            self.points_manager = parse_points(fileName, self.max_size_select.value(), self.max_size_select.value())
            self.max_size_select.setEnabled(True)
            self.min_size_select.setEnabled(True)
            self.plot3D.points_manager = self.points_manager
            self.start_simulation_btn.setText('Начать')
            self.update_plot()
        except Exception as e:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText(f'Не удалось загрузить файл:\"{fileName}\"')
            msg.setDetailedText('Error: ' + str(e) + '\n' + 'Traceback: ' + traceback.format_exc())
            msg.exec_()
            return
        
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText('Данные успешно загружены')
        msg.exec_()


    def update_plot(self):
        self.plot3D.update_points()


from qasync import QEventLoop, QThreadExecutor
import asyncio
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    ex = MainWindow()
    ex.show()

    with loop:
        sys.exit(loop.run_forever())