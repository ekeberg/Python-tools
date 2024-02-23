import sys
import numpy as np
import vtk
import vtk.util.numpy_support
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt
import argparse
from eke import hdf5_tools


SLIDER_MAX = 100


class MyInteractorStyle(vtk.vtkInteractorStyleRubberBandPick):
    def __init__(self, synced_interactors):
        self._moving = False
        self._synced_interactors = synced_interactors
        self.AddObserver("MouseMoveEvent", self.mouseMoveEvent)
        self.AddObserver("LeftButtonPressEvent", self.leftButtonPressEvent)
        self.AddObserver("LeftButtonReleaseEvent", self.leftButtonReleaseEvent)
        self.AddObserver("MouseWheelForwardEvent", self.mouseWheelForwardEvent)
        self.AddObserver("MouseWheelBackwardEvent", self.mouseWheelBackwardEvent)

    def _render_synced(self):
        for interactor in self._synced_interactors:
            interactor.GetRenderWindow().Render()

    def leftButtonPressEvent(self, obj, event):
        self._moving = True
        self.OnLeftButtonDown()

    def leftButtonReleaseEvent(self, obj, event):
        self._moving = False
        self.OnLeftButtonDown()

    def mouseWheelForwardEvent(self, obj, event):
        self.OnMouseWheelForward()
        self._render_synced()

    def mouseWheelBackwardEvent(self, obj, event):
        self.OnMouseWheelBackward()
        self._render_synced()
        
    def mouseMoveEvent(self, obj, event):
        if self._moving:
            self.OnMouseMove()
            self._render_synced()


class LinkedSliders(QtWidgets.QWidget):
    def __init__(self, slider1, slider2):
        super().__init__()
        self.slider1 = slider1
        self.slider2 = slider2

        self.linked = True
        self.ratio = 1
        self.temporary_block = False

        self.slider1.valueChanged.connect(self.slider_updated)
        self.slider2.valueChanged.connect(self.slider_updated)

    def set_linked(self, linked):
        self.linked = linked
        if self.linked:
            self.ratio = self.slider1.value() / self.slider2.value()
            print(self.ratio)

    def slider_updated(self, value):
        sender = self.sender()
        if self.linked and not self.temporary_block:
            self.temporary_block = True
            if sender == self.slider1:
                print("slider1", value, self.ratio, int(value / self.ratio))
                self.slider2.setValue(int(value / self.ratio))
            if sender == self.slider2:
                print("slider2", value, self.ratio, int(value * self.ratio))
                self.slider1.setValue(int(value * self.ratio))
            self.temporary_block = False


class IsosurfaceViewer(QtWidgets.QMainWindow):
    def __init__(self, array1, array2):
        super(IsosurfaceViewer, self).__init__()

        # Create main widget
        self.main_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.main_widget)
        
        self.array1_max = array1.max()
        self.array2_max = array2.max()

        # Create VTK renderers
        self.vtk_widget1, self.contour1 = self.create_vtk_widget(array1)
        self.vtk_widget2, self.contour2 = self.create_vtk_widget(array2)

        self.linked_style1 = MyInteractorStyle([self.vtk_widget2])
        self.linked_style2 = MyInteractorStyle([self.vtk_widget1])

        self.unlinked_style1 = vtk.vtkInteractorStyleTrackballActor()
        self.unlinked_style2 = vtk.vtkInteractorStyleTrackballActor()

        self.vtk_widget1.SetInteractorStyle(self.linked_style1)
        self.vtk_widget2.SetInteractorStyle(self.linked_style2)

        plot1_layout = QtWidgets.QVBoxLayout()
        plot2_layout = QtWidgets.QVBoxLayout()

        plot1_layout.addWidget(self.vtk_widget1)
        plot2_layout.addWidget(self.vtk_widget2)

        # Create and connect slider
        self.slider1 = QtWidgets.QSlider(Qt.Horizontal)
        self.slider1.setRange(0, SLIDER_MAX)
        self.slider1.valueChanged.connect(self.update_isosurface1_level)
        plot1_layout.addWidget(self.slider1)

        self.slider2 = QtWidgets.QSlider(Qt.Horizontal)
        self.slider2.setRange(0, SLIDER_MAX)
        self.slider2.valueChanged.connect(self.update_isosurface2_level)
        plot2_layout.addWidget(self.slider2)

        self.slider1.setValue(50)  # Initial isosurface level
        self.slider2.setValue(50)  # Initial isosurface level

        self.slider_link = LinkedSliders(self.slider1, self.slider2)

        plots_layout = QtWidgets.QHBoxLayout()
        plots_layout.addLayout(plot1_layout)
        plots_layout.addLayout(plot2_layout)

        self.camera_checkbox = QtWidgets.QCheckBox("Sync cameras")
        self.camera_checkbox.setChecked(True)
        self.camera_checkbox.stateChanged.connect(self.toggle_link_cameras)

        self.level_checkbox = QtWidgets.QCheckBox("Sync isosurface level")
        self.level_checkbox.setChecked(True)
        self.level_checkbox.stateChanged.connect(self.slider_link.set_linked)
        
        layout = QtWidgets.QVBoxLayout(self.main_widget)
        layout.addLayout(plots_layout)
        layout.addWidget(self.camera_checkbox)
        layout.addWidget(self.level_checkbox)


        # Link the camera of the two VTK views
        self.link_cameras()

    def create_vtk_widget(self, array):
        # Create VTK widget
        vtk_widget = QVTKRenderWindowInteractor(self.main_widget)
        renderer = vtk.vtkRenderer()
        vtk_widget.GetRenderWindow().AddRenderer(renderer)

        # Create isosurface
        data = self.numpy_to_vtk_data(array)
        contour = vtk.vtkContourFilter()
        contour.SetInputData(data)
        contour.SetValue(0, 0.5*array.max())  # Initial isosurface level

        # Create a mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(contour.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Add actor to the renderer
        renderer.AddActor(actor)
        renderer.ResetCamera()

        return vtk_widget, contour

    def numpy_to_vtk_data(self, array):
        # Convert NumPy array to VTK data format
        flat_array = array.flatten()
        vtk_data_array = vtk.util.numpy_support.numpy_to_vtk(num_array=flat_array, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_data = vtk.vtkImageData()
        vtk_data.SetDimensions(array.shape)
        vtk_data.GetPointData().SetScalars(vtk_data_array)
        return vtk_data

    def update_isosurface1_level(self, value):
        mapped_value = value / SLIDER_MAX * self.array1_max
        self.contour1.SetValue(0, mapped_value)
        self.vtk_widget1.GetRenderWindow().Render()

    def update_isosurface2_level(self, value):
        mapped_value = value / SLIDER_MAX * self.array2_max
        self.contour2.SetValue(0, mapped_value)
        self.vtk_widget2.GetRenderWindow().Render()

    def link_cameras(self):
        camera1 = self.vtk_widget1.GetRenderWindow().GetRenderers().GetFirstRenderer().GetActiveCamera()
        self.vtk_widget2.GetRenderWindow().GetRenderers().GetFirstRenderer().SetActiveCamera(camera1)

    def toggle_link_cameras(self, state):
        if state:
            self.vtk_widget1.SetInteractorStyle(self.linked_style1)
            self.vtk_widget2.SetInteractorStyle(self.linked_style2)
        else:
            self.vtk_widget1.SetInteractorStyle(self.unlinked_style1)
            self.vtk_widget2.SetInteractorStyle(self.unlinked_style2)


def main():
    parser = argparse.ArgumentParser(description="Plot isosurfaces of two 3D maps.")
    parser.add_argument("file1", type=str, help="First hdf5 file and location")
    parser.add_argument("file2", type=str, help="Second hdf5 file and location")
    args = parser.parse_args()
    
    try:
        file1, key1 = hdf5_tools.parse_name_and_key(args.file1)
    except IOError:
        print("You must specify a dataset to plot. Available 3D datasets in file 1 are:")
        print("\n".join(hdf5_tools.list_datasets(file1, dimensions=3)))
        exit(1)

    try:
        file2, key2 = hdf5_tools.parse_name_and_key(args.file2)
    except IOError:
        print("You must specify a dataset to plot. Available 3D datasets in file 2 are:")
        print("\n".join(hdf5_tools.list_datasets(file2, dimensions=3)))
        exit(1)

    data1 = hdf5_tools.read_dataset(file1, key1)
    data2 = hdf5_tools.read_dataset(file2, key2)

    app = QtWidgets.QApplication(["plot_image_3d"])
    program = IsosurfaceViewer(data1, data2)
    program.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()