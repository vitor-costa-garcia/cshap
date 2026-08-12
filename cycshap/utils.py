from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Constantes compartilhadas
# ---------------------------------------------------------------------------

DEFAULT_CMAP = LinearSegmentedColormap.from_list("shap_official", ["#008bfb", "#ff0051"])

# Cores do espectro de fase usadas em sct()
COR_ANTI_HORARIO = "#FF0051"  # Avanço de fase (tangencial puro, sentido +)
COR_HORARIO = "#008BFB"       # Recuo de fase (tangencial puro, sentido -)
COR_RADIAL = "#888888"        # Impacto puramente de amplitude (radial puro)

_QUADRANT_POSITIONS = {
    "top_right": (0.96, 0.96, "right", "top"),
    "top_left": (0.04, 0.96, "left", "top"),
    "bottom_left": (0.04, 0.04, "left", "bottom"),
    "bottom_right": (0.96, 0.04, "right", "bottom"),
}

# ---------------------------------------------------------------------------
# Helpers internos (compartilhados entre as funções públicas)
# ---------------------------------------------------------------------------

def _add_quadrant_labels(ax: plt.Axes, quadrant_labels: dict[str, str] | None) -> None:
    """Desenha rótulos nos 4 cantos do Axes, usado por scs() e sct()."""
    if not quadrant_labels:
        return
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5, alpha=0.8)
    for key, text in quadrant_labels.items():
        if key in _QUADRANT_POSITIONS and text:
            x, y, ha, va = _QUADRANT_POSITIONS[key]
            ax.text(
                x, y, text,
                transform=ax.transAxes,
                ha=ha, va=va,
                fontsize=10,
                fontweight="bold",
                color="#333333",
                bbox=bbox_props,
                zorder=10,
            )


def _finalize(ax: plt.Axes, owns_fig: bool, show: bool, save_path: str | Path | None) -> plt.Axes:
    """Salva/exibe/fecha a figura de forma padronizada e retorna o Axes."""
    if save_path:
        ax.figure.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
        if owns_fig:
            plt.close(ax.figure)
    # se show=False: não fecha nada — o chamador ainda pode acessar/editar ax.figure
    return ax


def _blend_color(cor_extrema, cor_neutra, peso):
    """peso=0 -> cor_neutra (radial); peso=1 -> cor_extrema (tangencial puro)."""
    rgb_extrema = np.array(plt.matplotlib.colors.to_rgb(cor_extrema))
    rgb_neutra = np.array(plt.matplotlib.colors.to_rgb(cor_neutra))
    return tuple(rgb_neutra + (rgb_extrema - rgb_neutra) * peso)