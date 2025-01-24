from himena import StandardType
from himena.plugins import configure_submenu

MENUS_TEST = ["tools/stats-test", "/model_menu/stats-test"]
MENUS_DIST = ["tools/stats-dist", "/model_menu"]
configure_submenu(MENUS_TEST, title="Statistical Tests")
configure_submenu(MENUS_DIST, title="Distributions")

TABLE_LIKE = [StandardType.TABLE, StandardType.DATAFRAME, StandardType.EXCEL]
