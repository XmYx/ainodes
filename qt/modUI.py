import sys, math
import PyQt6.QtCore as QtCore
import PyQt6.QtWidgets as QtWidgets
import PyQt6.QtGui as QtGui
from PyQt6.QtWidgets import QListView


# ---------------------------------------------------------------------------------------------------------
class clsFilePreviewModel(QtGui.QFileSystemModel):
    def __init__(self, cacheWidth=100, cacheHeight=100):
        super().__init__()
        self.previews = {'None': None}
        self.cacheWidth = cacheWidth
        self.cacheHeight = cacheHeight
        self.ncols = 2

    #----------------------------------------------------------------------------
    def getPreview(self, index):
        itemName = super().data(index, QtCore.Qt.DisplayRole)

        if itemName not in self.previews:
            qpm = QtGui.QPixmap(self.rootPath() + "/" + itemName)

            if qpm is None or qpm.isNull():
                qpm = super().data(index, QtCore.Qt.DecorationRole)
                if qpm and not qpm.isNull():
                    qpm = qpm.pixmap(self.cacheWidth, self.cacheHeight)
            if qpm and not qpm.isNull():
                qpm = qpm.scaled(self.cacheWidth, self.cacheHeight, QtCore.Qt.KeepAspectRatio)

            self.previews[itemName] = qpm
        return self.previews[itemName]

    # ----------------------------------------------------------------------------
    def data(self, index, role):
        if role == QtCore.Qt.DecorationRole:
            return self.getPreview(index)
        else:
            return super().data(index, role)

# ---------------------------------------------------------------------------------------------------------
class clsIconsGrid(QtWidgets.QWidget):
    def __init__(self, dir_path):
        super().__init__()

        self.gridSize = QtCore.QSize(200,200)

        self.path = dir_path
        self.files = clsFilePreviewModel()
        self.files.setRootPath(self.path)

        self.view = QtWidgets.QListView()
        self.view.setModel(self.files)
        self.view.setRootIndex(self.files.index(dir_path))
        self.view.setViewMode(QListView.ViewMode.IconMode)
        self.view.setGridSize(self.gridSize)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.view)
        self.setLayout(layout)