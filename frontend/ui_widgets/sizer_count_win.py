from PyQt6 import uic
from PyQt6.QtWidgets import QWidget

class SizerCountWin(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi('creator/sizer_count.ui', self)
