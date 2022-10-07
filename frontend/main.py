from frontend.main.main_window import Ui_MainWindow
from frontend.ui_widgets.test_widget import Ui_Form
from frontend.ui_widgets.txt2img_params import Ui_txt2img_params

from frontend.ui_widgets.animation import Animation

from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PIL import Image
import torch

import sys, traceback, time
from PyQt6 import QtCore
from PyQt6 import QtWidgets as qtw
from PyQt6 import QtCore as qtc
from PyQt6.QtGui import QIcon, QPixmap

from diffusers import StableDiffusionPipeline

from ldm.generate import Generate
gr = Generate(  weights     = 'models/sd-v1-4.ckpt',
                config     = 'configs/stable-diffusion/v1-inference.yaml',
                )

from backend.singleton import singleton

import backend.settings as settings
settings.load_settings_json()

gs = singleton

class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    progress
        int indicating % progress

    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done

class GenerateWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

        #self.ui = Ui_Form()
        self.mw = qtw.QMainWindow()
        self.ui = Ui_MainWindow()
        self.test = Ui_Form()
        self.txt2img = Ui_txt2img_params()


        self.ui.setupUi(self)
        self.test.setupUi(self)
        self.txt2img.setupUi(self)

        #self.mw.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.txt2img.dockWidget)
        #print(self.dockWidget.preview.scene())




        #self.show()
        #self.txt2img = self.ui.dockWidget
        #self.ui.preview = self.ui.dockWidget.preview
        self.test.scene = QGraphicsScene()
        self.test.graphicsView.setScene(self.test.scene)

        #self.ui.actionText_2_Image.triggered.connect(self.txt2img.show)
        self.test.load_btn.clicked.connect(self.test1)
        #self.ui.pushButton.clicked.connect(self.generate)

    def run_txt2img(self, progress_callback):

        results = gr.prompt2image(prompt   = "an astronaut riding a horse",
                                    outdir   = "./outputs/")
        for row in results:
            print(f'filename={row[0]}')
            print(f'seed    ={row[1]}')
            output = f'outputs/sample.png'
            row[0].save(output)
            self.image_path = output
    def test1(self):
        self.image_path = f'outputs/sample.png'
        self.get_pic()



    def txt2img_thread(self):
        # Pass the function to execute
        worker = Worker(self.run_txt2img) # Any other args, kwargs are passed to the run function
        worker.signals.result.connect(self.get_pic)

        # Execute
        self.threadpool.start(worker)
    def get_pic(self): #from self.image_path


        self.image_qt = QImage(self.image_path)

        pic = QGraphicsPixmapItem()
        pic.setPixmap(QPixmap.fromImage(self.image_qt))

        #pixmap = QPixmap(self.image_path)
        self.test.scene.addItem(pic)
        #self.ui.label.setPixmap(pixmap)

if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)

    widget = GenerateWindow()
    widget.show()
    sys.exit(app.exec())