"""Small GUI tools for use in smaller python scripts. I think the implementation
of the Manipulator looks weirt, but it works. Needs non integer manipulation."""
from QtVersions import QtCore, QtGui
import sys

class Manipulator(QtGui.QMainWindow):
    """Run a function with varying numerical input using a slider to change the value.
    Designed for use with matplotlib."""
    def __init__(self, function, value_ranges, value_names, parent=None):
        super(Manipulator, self).__init__(parent)
        self.setWindowTitle("Manipulator")
        self._function = function
        self._ranges = value_ranges
        self._values = [v[0] for v in self._ranges]
        self._value_names = value_names
        self._number_of_variables = len(self._ranges)
        if self._value_names and len(self._value_names) != self._number_of_variables:
            raise ValueError("Wrong number of variable names.")
        self._manipulate_functions = []
        self._create_window()

    class _VariableChanger(object):
        """Stores the function and calles it on request."""
        def __init__(self, function, variable_index, values_pointer):
            self._function = function
            self._variable_index = variable_index
            self._values_pointer = values_pointer

        def change_this_signal(self, value):
            """Call the function again with the given input"""
            self._values_pointer[self._variable_index] = value
            self._function(*tuple(self._values_pointer))

    def _create_window(self):
        """Contains all the Qt stuff to create a window and connect sliders to
        _VariableChanger objects."""
        self.window = QtGui.QWidget()

        hbox = QtGui.QHBoxLayout()
        for variable in range(self._number_of_variables):
            slider = QtGui.QSlider(QtCore.Qt.Vertical)
            slider.setMinimum(self._ranges[variable][0])
            slider.setMaximum(self._ranges[variable][1])
            slider.setValue(self._values[variable])

            self._manipulate_functions.append(self._VariableChanger(self._function, variable, self._values))
            # self.connect(slider, SIGNAL('sliderMoved(int)'), self._manipulate_functions[variable].change_this_signal)
            # self.connect(slider, SIGNAL('valueChanged(int)'), self._manipulate_functions[variable].change_this_signal)
            slider.sliderMoved.connect(self._manipulate_functions[variable].change_this_signal)
            slider.valueChanged.connect(self._manipulate_functions[variable].change_this_signal)

            title_label = QtGui.QLabel(self._value_names[variable])
            min_label = QtGui.QLabel(str(self._ranges[variable][0]))
            max_label = QtGui.QLabel(str(self._ranges[variable][1]))

            vbox = QtGui.QVBoxLayout()
            vbox.addWidget(title_label)
            vbox.addWidget(max_label)
            vbox.addWidget(slider)
            vbox.addWidget(min_label)
            hbox.addLayout(vbox)

        self.window.setLayout(hbox)
        self.setCentralWidget(self.window)

    # class PlotThread(object):
    #     def __init__(self, function, value):
    #         self.function = function
    #         self.value = value

    #     def run(self):
    #         self.function(self.value)

    # def start(self):
    #     #p = self.PlotThread(self.function, self.ranges[0])
    #     #self.t = Thread(target=p)
    #     #self.function(self.ranges[0])
    #     pass

# class ProgramThread(Thread):
#     def __init__(self, function, ranges):
#         self._function = function
#         self._ranges = ranges
#         self._values = [(r[0]+r[1])/2. for r in self._ranges]
#         self.functions
#     def run():
#         self._function(*self.values)

def manipulate(function, value_range, value_name=None):
    """Manipulate a single variable."""
    if not value_name:
        value_name = ['Variable']
    app = QtGui.QApplication(sys.argv)
    manipulator = Manipulator(function, [value_range], [value_name])
    manipulator.show()
    #manipulator.start()
    app.exec_()

def manipulate_multi(function, value_ranges, value_names=None):
    """Manipulate multiple variables at the same time."""
    app = QtGui.QApplication(sys.argv)
    if not value_names:
        value_names = [str(i) for i in range(len(value_ranges))]
    manipulator = Manipulator(function, value_ranges, value_names)
    manipulator.show()
    #manipulator.start()
    app.exec_()
