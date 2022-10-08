from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6 import QtWidgets as qtw
import sys
app = qtw.QApplication(sys.argv)
pixmap = QPixmap('frontend/main/splash.png')
splash = QSplashScreen(pixmap)
splash.show()



import random

from ui_classes import *
from nodeeditor.utils import loadStylesheet
from nodeeditor.node_editor_window import NodeEditorWindow
from qtpy.QtWidgets import QApplication as qapp


from PyQt6 import uic
from PyQt6.QtCore import *



import traceback, time
from PyQt6 import QtCore

from PyQt6 import QtCore as qtc
from PyQt6.QtGui import QIcon, QPixmap

from ldm.generate import Generate

gr = Generate(  weights     = 'models/sd-v1-4.ckpt',
                config     = 'configs/stable-diffusion/v1-inference.yaml',
                )

from backend.singleton import singleton

import backend.settings as settings
settings.load_settings_json()


gs = singleton

gs.result = ""

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
        self.image_path = ""

        #print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

        uic.loadUi("frontend/main/main_window.ui", self)

        self.home()
    def home(self):

        self.preview = Preview()
        self.sizer_count = SizerCount()
        self.sampler = Sampler()
        self.runner = Runner()
        self.anim = Anim()
        self.prompt = Prompt()
        self.thumbnails = Thumbnails()

        #app2  = qapp(sys.argv)
        #self.nodes = NodeEditorWindow()
        #self.nodes.nodeeditor.addNodes()

        self.thumbnails.thumbs.itemClicked.connect(self.viewImageClicked)
        self.thumbnails.thumbs.addItem(QListWidgetItem(QIcon('frontend/main/splash.png'), "Earth"))

        self.preview.scene = QGraphicsScene()
        self.preview.graphicsView.setScene(self.preview.scene)

        self.sizer_count.heightNumber.display(str(self.sizer_count.heightSlider.value()))
        self.sizer_count.widthNumber.display(str(self.sizer_count.widthSlider.value()))
        self.sizer_count.samplesNumber.display(str(self.sizer_count.samplesSlider.value()))
        self.sizer_count.batchSizeNumber.display(str(self.sizer_count.batchSizeSlider.value()))
        self.sizer_count.stepsNumber.display(str(self.sizer_count.stepsSlider.value()))
        self.sizer_count.scaleNumber.display(str(self.sizer_count.scaleSlider.value()))


        self.setCentralWidget(self.preview)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.sizer_count)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.sampler)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.runner)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.anim)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.prompt)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.thumbnails)

    def viewImageClicked(self, item):
        #self.preview.setPixmap(item.image())
        #pic = QImage(item[0])
        imageSize = item.icon().actualSize(QSize(512, 512))

        pic = QGraphicsPixmapItem()
        pic.setPixmap(item.icon().pixmap(imageSize))


        self.preview.scene.addItem(pic)

    def run_txt2img(self, progress_callback):
        results = gr.prompt2image(prompt   = self.prompt.textEdit.toPlainText(),
                                  outdir   = "./outputs/")
        for row in results:
            print(f'filename={row[0]}')
            print(f'seed    ={row[1]}')
            filename = random.randint(10000, 99999)
            output = f'outputs/{filename}.png'
            row[0].save(output)
            self.thumbnails.thumbs.addItem(QListWidgetItem(QIcon(output), "Earth"))
        self.image_path = output



    def txt2img_thread(self):
        # Pass the function to execute
        worker = Worker(self.run_txt2img) # Any other args, kwargs are passed to the run function
        worker.signals.result.connect(self.get_pic)
        # Execute
        threadpool.start(worker)
    def get_pic(self): #from self.image_path
        print("ok")
        image_qt = QImage(self.image_path)

        pic = QGraphicsPixmapItem()
        pic.setPixmap(QPixmap.fromImage(image_qt))

        pixmap = QPixmap(self.image_path)
        self.preview.scene.addItem(pic)
def update(target, value):
    print(type(target))
    print(type(value))
    target.setText(str(value.value()))

if __name__ == "__main__":

    mainWindow = GenerateWindow()
    threadpool = QThreadPool()
    mainWindow.show()
    app2 = qapp(sys.argv)
    wnd = NodeEditorWindow()
    wnd.nodeeditor.addNodes()

    splash.finish(mainWindow)

    mainWindow.runner.runButton.clicked.connect(mainWindow.txt2img_thread)

    sys.exit(app.exec())
