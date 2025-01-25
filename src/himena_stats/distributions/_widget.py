from __future__ import annotations

from typing import TYPE_CHECKING
from qtpy import QtWidgets as QtW, QtCore, QtGui
from himena import WidgetDataModel, StandardType
from himena.plugins import validate_protocol
from himena_stats.distributions._utils import draw_pdf_or_pmf, infer_edges

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
        xlow, xhigh = infer_edges(dist)
        x, y = draw_pdf_or_pmf(dist, xlow, xhigh)
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
        super().__init__(QtCore.Qt.Orientation.Vertical)
        self._img_view = QDistGraphics()
        self._param_view = QDistParameters()
        self._dist: stats.rv_frozen | None = None
        self.addWidget(self._img_view)
        self.addWidget(self._param_view)
        self.setSizes([100, 60])

    @validate_protocol
    def update_model(self, model: WidgetDataModel):
        dist: stats.rv_frozen = model.value
        if dist.args:
            raise NotImplementedError
        self._img_view.set_dist(dist)
        self._param_view.set_dist(dist)
        self._dist = dist

    @validate_protocol
    def to_model(self) -> WidgetDataModel:
        if self._dist is None:
            raise ValueError("No distribution is set.")
        return WidgetDataModel(
            value=self._dist,
            type=self.model_type(),
        )

    @validate_protocol
    def model_type(self) -> str:
        return StandardType.DISTRIBUTION

    @validate_protocol
    def size_hint(self) -> tuple[int, int]:
        return 220, 200
