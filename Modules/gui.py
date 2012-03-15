from PySide.QtCore import *
from PySide.QtGui import *
from threading import Thread
import sys

class Manipulator(QMainWindow):
    def __init__(self, function, value_ranges, value_names, parent=None):
        QMainWindow.__init__(self, parent)
        self.setWindowTitle("Manipulator")
        self.function = function
        self.ranges = value_ranges
        self._value_names = value_names
        self.number_of_variables = len(self.ranges)
        if (self._value_names and len(self._value_names) != self.number_of_variables): raise InputError("Wrong number of variable names.")
        self.create_window(value_ranges)

    class VariableChanger(object):
        def __init__(self, function, variable_index, values_pointer):
            self._function = function
            self._variable_index = variable_index
            self._values_pointer = values_pointer
        def change_this_signal(self, value):
            self._values_pointer[self._variable_index] = value
            self._function(*tuple(self._values_pointer))

    def create_window(self, ranges):
        self.window = QWidget()
        self.values = [v[0] for v in ranges]

        self._manipulate_functions = []
        hbox = QHBoxLayout()
        for variable in range(self.number_of_variables):
            self.slider = QSlider(Qt.Vertical)
            self.slider.setMinimum(self.ranges[variable][0])
            self.slider.setMaximum(self.ranges[variable][1])
            self.slider.setValue(self.values[variable])

            self._manipulate_functions.append(self.VariableChanger(self.function, variable, self.values))
            self.connect(self.slider, SIGNAL('sliderMoved(int)'), self._manipulate_functions[variable].change_this_signal)
            self.connect(self.slider, SIGNAL('valueChanged(int)'), self._manipulate_functions[variable].change_this_signal)

            self.title_label = QLabel(self._value_names[variable])
            self.min_label = QLabel(str(self.ranges[variable][0]))
            self.max_label = QLabel(str(self.ranges[variable][1]))

            vbox = QVBoxLayout()
            vbox.addWidget(self.title_label)
            vbox.addWidget(self.max_label)
            vbox.addWidget(self.slider)
            vbox.addWidget(self.min_label)
            hbox.addLayout(vbox)
        
        self.window.setLayout(hbox)
        self.setCentralWidget(self.window)

    class PlotThread(object):
        def __init__(self, function, value):
            self.function = function
            self.value = value

        def run(self):
            self.function(self.value)

    def start(self):
        #p = self.PlotThread(self.function, self.ranges[0])
        #self.t = Thread(target=p)
        #self.function(self.ranges[0])
        pass

class ProgramThread(Thread):
    def __init__(self, function, ranges):
        self._function = function
        self._ranges = ranges
        self._values = [(r[0]+r[1])/2. for r in self._ranges]
        self.functions
    def run():
        self._function(*self.values)

def manipulate(function, value_range, value_name=None):
    if not value_name:
        value_name = ['Variable']
    app = QApplication(sys.argv)
    manipulator = Manipulator(function, (value_range), value_name)
    manipulator.show()
    manipulator.start()
    app.exec_()
    
def manipulate_multi(function, value_ranges, value_names=None):
    app = QApplication(sys.argv)
    if not value_names:
        value_names = [str(i) for i in range(len(value_ranges))]
    manipulator = Manipulator(function, value_ranges, value_names)
    manipulator.show()
    manipulator.start()
    app.exec_()
