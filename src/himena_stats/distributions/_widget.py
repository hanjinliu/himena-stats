from __future__ import annotations

from typing import TYPE_CHECKING
from qtpy import QtWidgets as QtW, QtCore, QtGui
import numpy as np
from himena import WidgetDataModel, StandardType
from himena.plugins import validate_protocol

if TYPE_CHECKING:
    from scipy import stats


class QDistGraphics(QtW.QGraphicsView):
    """Graphics view for displaying a distribution."""

    def __init__(self):
        scene = QtW.QGraphicsScene()
        super().__init__(scene)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._color = QtGui.QColor(128, 128, 128)
        self.setTransform(QtGui.QTransform().scale(1, -1))  # upside down

    def set_dist(self, dist: stats.rv_frozen):
        scene = self.scene()
        scene.clear()
        xlow: float = dist.ppf(0.001)
        xhigh: float = dist.ppf(0.999)
        if hasattr(dist, "pdf"):  # contiuous
            x = np.linspace(xlow, xhigh, 100)
            y: np.ndarray = dist.pdf(x)
        elif hasattr(dist, "pmf"):  # discrete
            x0 = np.arange(xlow, xhigh + 1)
            y0: np.ndarray = dist.pmf(x0)
            x = np.repeat(np.concatenate([x0, [xhigh + 1]]) - 0.5, 3)[1:-1]
            y = np.concatenate([np.repeat(y0, 3), [0]])
            y[::3] = 0
        else:
            raise TypeError(f"Type {type(dist)} not allowed.")

        polygon = QtGui.QPolygonF([QtCore.QPointF(x[i], y[i]) for i in range(len(x))])
        scene.addPolygon(polygon, QtGui.QPen(self._color, 0), QtGui.QBrush(self._color))
        self.fit_item()

    def resizeEvent(self, event):
        self.fit_item()
        super().resizeEvent(event)

    def fit_item(self):
        self.fitInView(self.scene().itemsBoundingRect())


class QDistParameters(QtW.QPlainTextEdit):
    def __init__(self):
        super().__init__()
        self.setWordWrapMode(QtGui.QTextOption.WrapMode.NoWrap)
        self.setReadOnly(True)

    def set_dist(self, dist: stats.rv_frozen):
        dist_name = dist.dist.name
        params = [f"{k} = {v}" for k, v in dist.kwds.items()]
        newline = "\n".join(params)
        self.setPlainText(f"{dist_name}\n\n{newline}")


class QDistributionView(QtW.QSplitter):
    def __init__(self):
        super().__init__(QtCore.Qt.Orientation.Horizontal)
        self._img_view = QDistGraphics()
        self._param_view = QDistParameters()
        self._dist = None
        self.addWidget(self._img_view)
        self.addWidget(self._param_view)

    @validate_protocol
    def update_model(self, model: WidgetDataModel):
        dist: stats.rv_frozen = model.value
        self._img_view.set_dist(dist)
        self._param_view.set_dist(dist)

    @validate_protocol
    def to_model(self) -> WidgetDataModel:
        return WidgetDataModel(
            value=self._dist,
            type=self.model_type(),
        )

    @validate_protocol
    def model_type(self) -> str:
        return StandardType.DISTRIBUTION

    @validate_protocol
    def size_hint(self) -> tuple[int, int]:
        return 360, 200
