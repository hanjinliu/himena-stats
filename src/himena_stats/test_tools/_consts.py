from himena import StandardType
from himena.plugins import configure_submenu

MENUS = ["tools/stats", "/model_menu/stats"]
configure_submenu("tools/stats", title="Statistical Tests")
configure_submenu("/model_menu/stats", title="Statistical Tests")

TABLE_LIKE = [StandardType.TABLE, StandardType.DATAFRAME, StandardType.EXCEL]
