from PyQt6.QtWidgets import QApplication
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6 import QtWidgets as qtw
import sys
app = QApplication(sys.argv)
pixmap = QPixmap('frontend/main/splash.png')
splash = QSplashScreen(pixmap)
splash.show()
#from PyQt6.Qt import *
import transformers
from transformers import CLIPTokenizer, CLIPTextModel
from transformers import BertTokenizerFast
import warnings

import random

from ui_classes import *

#from nodeeditor.utils import loadStylesheet
#from nodeeditor.node_editor_window import NodeEditorWindow
#from frontend.example_calculator.calc_window import CalculatorWindow
#from qtpy.QtWidgets import QApplication as qapp


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

from backend.ui_func import getLatestGeneratedImagesFromPath

gs.album = getLatestGeneratedImagesFromPath()

def prepare_loading():
    transformers.logging.set_verbosity_error()

    # this will preload the Bert tokenizer fles
    print('preloading bert tokenizer...')

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    print('...success')

    # this will download requirements for Kornia
    print('preloading Kornia requirements (ignore the deprecation warnings)...')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        import kornia
    print('...success')

    version = 'openai/clip-vit-large-patch14'

    print('preloading CLIP model (Ignore the deprecation warnings)...')
    sys.stdout.flush()

    tokenizer = CLIPTokenizer.from_pretrained(version)
    transformer = CLIPTextModel.from_pretrained(version)
    print('\n\n...success')

    # In the event that the user has installed GFPGAN and also elected to use
    # RealESRGAN, this will attempt to download the model needed by RealESRGANer
    gfpgan = False
    try:
        from realesrgan import RealESRGANer

        gfpgan = True
    except ModuleNotFoundError:
        pass

    if gfpgan:
        print('Loading models from RealESRGAN and facexlib')
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper

            RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=2,
                ),
            )

            RealESRGANer(
                scale=4,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                model=RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=4,
                ),
            )

            FaceRestoreHelper(1, det_model='retinaface_resnet50')
            print('...success')
        except Exception:
            import traceback

            print('Error loading GFPGAN:')
            print(traceback.format_exc())

import platform

if "macOS" in platform.platform():
    gs.platform = "macOS"
    prepare_loading()

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
        self.load_history()
        #self.show_anim()

        self.actionAnim.triggered.connect(self.show_anim)
        self.actionPreview.triggered.connect(self.show_preview)
        self.actionPrompt.triggered.connect(self.show_prompt)
        self.actionRunControl.triggered.connect(self.show_runner)
        self.actionSampler.triggered.connect(self.show_sampler)
        self.actionSliders.triggered.connect(self.show_sizer_count)
        self.actionThumbnails.triggered.connect(self.show_thumbnails)

        #self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        #self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)

        #self.pix_map_item = self.preview.scene.addPixmap(self.pix_map)
        """self.global_factor = 1
        self.pix_map_item = QGraphicsPixmapItem()

    def scaleImage(self, factor):
        _pixmap = self.pic.scaledToHeight(int(factor*self.viewport().geometry().height()), Qt.SmoothTransformation)
        self.pix_map_item.setPixmap(_pixmap)
        self.preview.scene.setSceneRect(QRectF(_pixmap.rect()))

    def wheelEvent(self, event):
        factor = 1.5

        if QApplication.keyboardModifiers() == Qt.ControlModifier:
            view_pos = event.pos()
            scene_pos = self.mapToScene(view_pos)
            self.centerOn(scene_pos)

            if event.angleDelta().y() > 0 and self.global_factor < 20:
                self.global_factor *= factor
                self.scaleImage(self.global_factor)
            elif event.angleDelta().y() < 0 and self.global_factor > 0.2:
                self.global_factor /= factor
                self.scaleImage(self.global_factor)
        else:
            return super().wheelEvent(event)"""

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

        #wnd.show()

        self.thumbnails.thumbs.itemClicked.connect(self.viewImageClicked)
        #self.thumbnails.thumbs.addItem(QListWidgetItem(QIcon('frontend/main/splash.png'), "Earth"))



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
        #self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.anim)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.prompt)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.thumbnails)

        self.preview.scene = QGraphicsScene()
        self.preview.graphicsView.setScene(self.preview.scene)

        self.thumbnails.thumbsZoom.valueChanged.connect(self.updateThumbsZoom)




    def updateThumbsZoom(self):
        size = self.thumbnails.thumbsZoom.value()
        self.thumbnails.thumbs.setGridSize(QSize(size, size))
        self.thumbnails.thumbs.setIconSize(QSize(size, size))
    def update_scaleNumber(self):
        float = self.sizer_count.scaleSlider.value() / 1000
        self.sizer_count.scaleNumber.display(str(float))


    def show_anim(self):
        self.anim.show()
    def show_preview(self):
        self.preview.show()
    def show_prompt(self):
        self.prompt.show()
    def show_runner(self):
        self.runner.show()
    def show_sampler(self):
        self.sampler.show()
    def show_sizer_count(self):
        self.sizer_count.show()
    def show_thumbnails(self):
        self.thumbnails.show()

    def load_history(self):

        for image in gs.album:
            self.thumbnails.thumbs.addItem(QListWidgetItem(QIcon(image), str(image)))

    def viewImageClicked(self, item):
        #self.preview.setPixmap(item.image())
        #pic = QImage(item[0])
        imageSize = item.icon().actualSize(QSize(512, 512))

        self.preview.pic = QGraphicsPixmapItem()
        self.preview.pic.setPixmap(item.icon().pixmap(imageSize))


        self.preview.scene.addItem(self.preview.pic)
        #self.preview.graphicsView.setDragMode(QGraphicsView.ScrollHandDrag)

        #self.preview.graphicsView.setPhoto(item.icon().pixmap(imageSize))

    def run_txt2img(self, progress_callback):

        width=self.sizer_count.widthSlider.value()
        height=self.sizer_count.heightSlider.value()
        scale=self.sizer_count.scaleSlider.value()
        steps=self.sizer_count.stepsSlider.value()
        samples=self.sizer_count.samplesSlider.value()
        batchsize=self.sizer_count.batchSizeSlider.value()
        seamless=self.sampler.seamless.isChecked()
        full_precision=self.sampler.fullPrecision.isChecked()
        sampler=self.sampler.comboBox.currentText()

        print(sampler)


        """The full list of arguments to Generate() are:
        gr = Generate(
                  weights     = path to model weights ('models/ldm/stable-diffusion-v1/model.ckpt')
                  config     = path to model configuraiton ('configs/stable-diffusion/v1-inference.yaml')
                  iterations  = <integer>     // how many times to run the sampling (1)
                  steps       = <integer>     // 50
                  seed        = <integer>     // current system time
                  sampler_name= ['ddim', 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler', 'k_heun', 'k_lms', 'plms']  // k_lms
                  grid        = <boolean>     // false
                  width       = <integer>     // image width, multiple of 64 (512)
                  height      = <integer>     // image height, multiple of 64 (512)
                  cfg_scale   = <float>       // condition-free guidance scale (7.5)
                  )

"""

        all_images = []
        for i in range(batchsize):

            results = gr.prompt2image(prompt   = self.prompt.textEdit.toPlainText(),
                                      outdir   = "./outputs/",
                                      cfg_scale = scale,
                                      width  = width,
                                      height = height,
                                      iterations = samples,
                                      steps = steps,
                                      seamless = seamless,
                                      sampler = sampler,
                                      full_precision = full_precision)
            for row in results:
                print(f'filename={row[0]}')
                print(f'seed    ={row[1]}')
                filename = random.randint(10000, 99999)
                output = f'outputs/{filename}.png'
                row[0].save(output)
                self.image_path = output
                self.get_pic
                self.thumbnails.thumbs.addItem(QListWidgetItem(QIcon(output), str(self.prompt.textEdit.toPlainText())))
            #all_images.append(results)

            #return all_images








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
"""class CalculatorWin(CalculatorWindow):
    def __init__(self, *args, **kwargs):




        app2 = qapp(sys.argv)
        nodes = CalculatorWindow()
        nodes.show()
        app2.exec()"""
"""def show_nodes():
    #in main thread:
    CalculatorWin()

    #in a separate thread
    #worker = Worker(CalculatorWin) # Any other args, kwargs are passed to the run function
    # Execute
    #threadpool.start(worker)"""

if __name__ == "__main__":

    mainWindow = GenerateWindow()
    threadpool = QThreadPool()
    mainWindow.show()
    splash.finish(mainWindow)
    #mainWindow.thumbnails.setGeometry(680,0,800,600)

    mainWindow.runner.runButton.clicked.connect(mainWindow.txt2img_thread)
    #mainWindow.actionNodes.triggered.connect(show_nodes)
    mainWindow.sizer_count.scaleSlider.valueChanged.connect(mainWindow.update_scaleNumber)

    sys.exit(app.exec())
