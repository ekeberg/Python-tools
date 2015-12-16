import sys
import numpy
from eke.QtVersions import QtCore, QtGui
import vtk
from eke import vtk_tools
from eke import sphelper
from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from functools import partial
import argparse


class MainWindow(QtGui.QMainWindow):
    def __init__(self, volumes):
    #def __init__(self, volume_1, volume_2):    
        super(MainWindow, self).__init__()
        # volumes = [volume_1, volume_2]
        self._central_widget = QtGui.QWidget()

        self._volumes = volumes
        self._volume_max = [volume.max() for volume in self._volumes]
        
        self._vtk_widget = [QVTKRenderWindowInteractor(self._central_widget) for _ in range(len(volumes))]
        self._surface_generator = [vtk_tools.IsoSurface(volume, level=(self._value_calculator(0.5)*volume.max())) for volume in volumes]
            
        self._renderer = [vtk.vtkRenderer() for _ in range(len(volumes))]
        for renderer, vtk_widget in zip(self._renderer, self._vtk_widget):
            renderer.SetBackground(1., 1., 1.)
            vtk_widget.GetRenderWindow().AddRenderer(renderer)

        for surface_generator, renderer in zip(self._surface_generator, self._renderer):
            surface_generator.set_renderer(renderer)
            surface_generator.set_color((0.2, 0.8, 0.2))

        self._surface_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self._surface_slider.setTracking(True)
        self._slider_levels = 1000
        self._surface_slider.setRange(1, self._slider_levels)
        self._surface_slider.setValue(self._slider_levels/2)

        def on_slider_change(self, new_slider_value):
            new_surface_value = self._value_calculator(float(new_slider_value) / float(self._slider_levels))
            for surface_generator, volume_max in zip(self._surface_generator, self._volume_max):
                surface_generator.set_level(0, new_surface_value*volume_max)

        self._surface_slider.valueChanged.connect(partial(on_slider_change, self))
            
        plot_layout = QtGui.QHBoxLayout()
        for vtk_widget in self._vtk_widget:
            plot_layout.addWidget(vtk_widget)

        layout = QtGui.QVBoxLayout()
        layout.addLayout(plot_layout)
        layout.addWidget(self._surface_slider)

        vtk_tools.synchronize_renderers(self._renderer)

        self._central_widget.setLayout(layout)
        self.setCentralWidget(self._central_widget)

    @staticmethod
    def _value_calculator(ratio):
        return ratio
        
    def initialize(self):
        for vtk_widget in self._vtk_widget:
            vtk_widget.Initialize()
        for renderer in self._renderer:
            renderer.Render()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("volume", nargs="+")
    args = parser.parse_args()

    volumes = [sphelper.import_spimage(volume_file, ["image"]) for volume_file in args.volume]

    app = QtGui.QApplication(["eke_plot_multiple_3d.py"])
    program = MainWindow(volumes)
    program.show()
    program.initialize()
    sys.exit(app.exec_())
