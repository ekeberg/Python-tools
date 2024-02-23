#!/usr/bin/env python
import sys
from PyQt5 import QtCore, QtWidgets
import vtk
from eke import vtk_tools
from eke import sphelper
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("volume", nargs="+")
    args = parser.parse_args()

    volumes = [sphelper.import_spimage(volume_file, ["image"])
               for volume_file in args.volume]

    app = QtWidgets.QApplication(["eke_plot_multiple_3d.py"])
    program = MainWindow(volumes)
    program.show()
    program.initialize()
    sys.exit(app.exec_())


class ObjectInteractorStyle(vtk.vtkInteractorStyleTrackballActor):
    def __init__(self):
        self.AddObserver("MiddleButtonPressEvent",
                         self._do_nothing)
        self.AddObserver("MiddleButtonReleaseEvent",
                         self._do_nothing)
        self.AddObserver("RightButtonPressEvent",
                         self._do_nothing)
        self.AddObserver("RightButtonReleaseEvent",
                         self._do_nothing)
        # self.AddObserver("KeyPressEvent",
        #                  self._do_nothing)
        self.AddObserver("KeyReleaseEvent",
                         self._do_nothing)
        self.AddObserver("MouseWheelForwardEvent",
                         self._do_nothing)
        self.AddObserver("MouseWheelBackwardEvent",
                         self._do_nothing)

    def _do_nothing(self, obj, event):
        pass


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, volumes):
        super(MainWindow, self).__init__()
        self._central_widget = QtWidgets.QWidget()

        self._volumes = volumes
        self._volume_max = [volume.max() for volume in self._volumes]

        self._vtk_widget = [QVTKRenderWindowInteractor(self._central_widget)
                            for _ in range(len(volumes))]
        self._surface_generator = [
            vtk_tools.IsoSurface(
                volume, level=self._value_calculator(0.5)*volume.max())
            for volume in volumes]

        self._renderer = [vtk.vtkRenderer() for _ in range(len(volumes))]
        for renderer, vtk_widget in zip(self._renderer, self._vtk_widget):
            renderer.SetBackground(1., 1., 1.)
            vtk_widget.GetRenderWindow().AddRenderer(renderer)

        for surface_generator, renderer in zip(self._surface_generator,
                                               self._renderer):
            surface_generator.set_renderer(renderer)
            surface_generator.set_color((0.2, 0.8, 0.2))

        self._surface_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._surface_slider.setTracking(True)
        self._slider_levels = 1000
        self._surface_slider.setRange(1, self._slider_levels)
        self._surface_slider.setValue(self._slider_levels//2)

        self._surface_slider.valueChanged.connect(self.on_slider_change)

        self._checkbox = QtWidgets.QCheckBox("Free")
        self._checkbox.setTristate(False)
        self._checkbox.setCheckState(0)
        self._checkbox.stateChanged.connect(self.on_checkbox)

        plot_layout = QtWidgets.QHBoxLayout()
        for vtk_widget in self._vtk_widget:
            plot_layout.addWidget(vtk_widget)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(plot_layout)
        layout.addWidget(self._checkbox)
        layout.addWidget(self._surface_slider)

        self._synchronized_interactor_style = [
            vtk_tools.SynchronizedInteractorStyle()
            for _ in range(len(self._vtk_widget))]
        self._single_interactor_style = [
            ObjectInteractorStyle()
            for _ in range(len(self._vtk_widget))]

        # vtk_tools.synchronize_renderers(self._renderer)
        renderer_list = self._renderer
        for index, renderer in enumerate(renderer_list):
            render_window = renderer.GetRenderWindow()
            interactor = render_window.GetInteractor()
            # my_interactor_style = SynchronizedInteractorStyle()
            my_interactor_style = self._synchronized_interactor_style[index]
            interactor.SetInteractorStyle(my_interactor_style)
            my_interactor_style.add_renderer(renderer)
            for other_renderer in renderer_list:
                if other_renderer is not renderer:
                    my_interactor_style.add_renderer(other_renderer)
        camera = renderer_list[0].GetActiveCamera()
        for renderer in renderer_list:
            renderer.SetActiveCamera(camera)

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

    def on_slider_change(self, new_slider_value):
        new_surface_value = self._value_calculator(
            float(new_slider_value) / float(self._slider_levels))
        for surface_generator, volume_max in zip(self._surface_generator,
                                                 self._volume_max):
            surface_generator.set_level(0, new_surface_value*volume_max)

    def on_checkbox(self, state):
        print(f"checkbox: {state}")
        iterator = zip(self._synchronized_interactor_style,
                       self._single_interactor_style,
                       self._vtk_widget)
        if state:
            for synchronized_style, single_style, interactor in iterator:
                synchronized_style.SetInteractor(None)
                single_style.SetInteractor(interactor)
        else:
            for synchronized_style, single_style, interactor in iterator:
                single_style.SetInteractor(None)
                synchronized_style.SetInteractor(interactor)


if __name__ == "__main__":
    main()
