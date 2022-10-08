from PyQt6 import uic
from PyQt6.QtWidgets import QDockWidget, QGraphicsScene

class SizerCount(QDockWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("frontend/ui_widgets/sizer_count.ui", self)

class Sampler(QDockWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("frontend/ui_widgets/sampler.ui", self)

class Runner(QDockWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("frontend/ui_widgets/runner.ui", self)

class Prompt(QDockWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("frontend/ui_widgets/prompt.ui", self)

class Anim(QDockWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("frontend/ui_widgets/anim.ui", self)

class Preview(QDockWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("frontend/ui_widgets/preview.ui", self)
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)
