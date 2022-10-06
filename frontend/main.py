#from qt.main_ui import Ui_Form
from main_window import Ui_MainWindow
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PIL import Image
import torch

import sys, traceback, time

from PyQt6 import QtWidgets as qtw
from PyQt6 import QtCore as qtc
from PyQt6.QtGui import QIcon, QPixmap

from diffusers import StableDiffusionPipeline

from ldm.generate import Generate
gr = Generate(  weights     = 'models/sd-v1-4.ckpt',
                config     = 'configs/stable-diffusion/v1-inference.yaml',
                )
from backend.singleton import singleton
from backend.modelloader import load_models
from omegaconf import OmegaConf

import backend.settings as settings
settings.load_settings_json()

gs = singleton

cfg = "configs/stable-diffusion/v1-inference.yaml"
config = OmegaConf.load(cfg)
config.model.params.personalization_config.params.embedding_manager_ckpt = (
    ""
)
config.model.params.personalization_config.params.placeholder_tokens = (
    ""
)


from transformers import CLIPTokenizer, CLIPTextModel
import clip
from transformers import BertTokenizerFast
import sys
import transformers
import os
import warnings

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

        self.ui.setupUi(self)

        #self.show()
        self.txt2img = self.ui.dockWidget
        self.ui.actionText_2_Image.triggered.connect(self.txt2img.show)
        self.ui.pushButton.clicked.connect(self.oh_no)
        #self.ui.pushButton.clicked.connect(self.generate)
    def generate(self, progress_callback):
        gr = Generate(
            width=512,
            height=512,
            sampler_name="ddim",
            weights="models/sd-v1-4.ckpt",
            full_precision=False,
            config="configs/stable-diffusion/v1-inference.yaml",
            grid=False,
            # this is solely for recreating the prompt
            seamless=True,
            embedding_path=None,
            device_type='cuda',
            ignore_ctrl_c=False,
        )
        results = gr.prompt2image(prompt   = "an astronaut riding a horse",
                                    outdir   = "./outputs/")
        for row in results:
            print(f'filename={row[0]}')
            print(f'seed    ={row[1]}')
            output = f'outputs/sample.png'
            row[0].save(output)

        self.image_path = output
        #pixmap = QPixmap(output)
        #self.ui.label.setPixmap(pixmap)




    def generate_with_diffusers(self, progress_callback):

        #pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token="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")

        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            revision="fp16",
            torch_dtype=torch.float16,
            use_auth_token="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",
        )
        pipe = pipe.to("cuda")
        def dummy_checker(images, **kwargs): return images, False
        #pipe.safety_checker = dummy_checker

        prompt = "Astronaut riding a horse"
        #pipe.enable_attention_slicing()
        # First-time "warmup" pass (see explanation above)
        _ = pipe(prompt, num_inference_steps=2)

        # Results match those from the CPU device after the warmup pass.
        image = pipe(prompt, num_inference_steps=4).images[0]
        image.save("image.png")

    def get_pic(self):
        pixmap = QPixmap(self.image_path)
        self.ui.label.setPixmap(pixmap)

    def oh_no(self):
        # Pass the function to execute
        worker = Worker(self.generate) # Any other args, kwargs are passed to the run function
        worker.signals.result.connect(self.get_pic)

        # Execute
        self.threadpool.start(worker)




if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)

    widget = GenerateWindow()
    widget.show()
    sys.exit(app.exec())