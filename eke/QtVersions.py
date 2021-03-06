"""Allows loading of Qt without caring wether PyQt4 or pyside is installed."""
import sys
import os

default_variant = 'PyQt5'

env_api = os.environ.get('QT_API', 'pyqt')
if '--pyside' in sys.argv:
    variant = 'PySide'
elif '--pyqt4' in sys.argv:
    variant = 'PyQt4'
elif '--pyqt5' in sys.argv:
    variant = 'PyQt5'
elif env_api == 'pyside':
    variant = 'PySide'
elif env_api == 'pyqt':
    variant = 'PyQt5'
else:
    variant = default_variant

if variant == 'PySide':
    from PySide import QtGui, QtCore, QtOpenGL
    # This will be passed on to new versions of matplotlib
    os.environ['QT_API'] = 'pyside'
    def QtLoadUI(uifile):
        from PySide import QtUiTools
        loader = QtUiTools.QUiLoader()
        uif = QtCore.QFile(uifile)
        uif.open(QtCore.QFile.ReadOnly)
        result = loader.load(uif)
        uif.close()
        return result
elif variant == 'PyQt4':
    import sip
    api2_classes = [
        'QData', 'QDateTime', 'QString', 'QTextStream',
        'QTime', 'QUrl', 'QVariant',
    ]
    for cl in api2_classes:
        try:
            sip.setapi(cl, 2)
        except ValueError:
            sip.setapi(cl, 1)
    from PyQt4 import QtGui, QtCore, QtOpenGL
    QtCore.Signal = QtCore.pyqtSignal
    QtCore.Slot = QtCore.pyqtSlot
    QtCore.QString = str
    os.environ['QT_API'] = 'pyqt'
    def QtLoadUI(uifile):
        from PyQt4 import uic
        return uic.loadUi(uifile)
elif variant == 'PyQt5':
    import sip
    api2_classes = [
        'QData', 'QDateTime', 'QString', 'QTextStream',
        'QTime', 'QUrl', 'QVariant',
    ]
    for cl in api2_classes:
        try:
            sip.setapi(cl, 2)
        except ValueError:
            sip.setapi(cl, 1)
    from PyQt5 import QtGui, QtCore, QtOpenGL, QtWidgets
    QtCore.Signal = QtCore.pyqtSignal
    QtCore.Slot = QtCore.pyqtSlot
    QtCore.QString = str
    os.environ['QT_API'] = 'pyqt'
    def QtLoadUI(uifile):
        from PyQt5 import uic
        return uic.loadUi(uifile)
else:
    raise ImportError("Python Variant not specified")

__all__ = [QtGui, QtCore, QtLoadUI, QtWidgets, variant]
