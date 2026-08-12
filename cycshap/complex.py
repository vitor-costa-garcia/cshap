"""
Visualizações SHAP para alvos representados no plano complexo (Argand-Gauss),
onde cada explicação é um número complexo (parte real + parte imaginária).

Segue o mesmo contrato do módulo `cycshap.circular`:
- Recebem os dados SHAP e, quando aplicável, um Axes opcional (`ax`).
- Retornam sempre o `plt.Axes` utilizado.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Importamos as funções e constantes compartilhadas do módulo circular
# (Nota: Em refatorações futuras, você pode mover isso para um utils.py)
from .circular import bar_mag
from .utils import (
    _finalize,
    _blend_color,
    DEFAULT_CMAP,
    COR_ANTI_HORARIO,
    COR_HORARIO,
    COR_RADIAL
)
# ---------------------------------------------------------------------------
# Funções públicas
# ---------------------------------------------------------------------------

def scs_complex(
    shap_complex: np.ndarray,
    feature_values: np.ndarray | pd.Series,
    feature_name: str = "Feature",
    cmap: str | LinearSegmentedColormap = DEFAULT_CMAP,
    show_polar_grid: bool = True,
    ax: plt.Axes | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """Gera o gráfico SHAP 2D no Plano Complexo (Argand-Gauss) com iso-magnitudes."""
    shap_real = np.real(shap_complex)
    shap_imag = np.imag(shap_complex)

    owns_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 8))

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

    max_val = max(np.abs(shap_real).max(), np.abs(shap_imag).max()) * 1.15
    max_val = max_val if max_val > 0 else 1.0
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    if show_polar_grid:
        mags = np.linspace(max_val * 0.25, max_val * 0.75, 3)
        for m in mags:
            circle = plt.Circle((0, 0), m, color="gray", linestyle=":", fill=False, alpha=0.4)
            ax.add_patch(circle)
            ax.text(
                m * np.cos(np.pi / 4), m * np.sin(np.pi / 4),
                f"|SHAP|={m:.2f}", fontsize=7, color="gray", alpha=0.8,
            )

    scatter = ax.scatter(
        shap_real, shap_imag, c=feature_values, cmap=cmap,
        alpha=0.7, edgecolors="none", s=40, zorder=3,
    )

    ax.set_xlabel("Impacto na Parte Real $\\text{Re}(\\phi)$ — (Amplitude / Em Fase)")
    ax.set_ylabel("Impacto na Parte Imaginária $\\text{Im}(\\phi)$ — (Quadratura / Fase)")
    ax.set_title(f"Dispersão SHAP no Plano Complexo: {feature_name}")

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(f"Valor Real de {feature_name}")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    return _finalize(ax, owns_fig, show, save_path)


def sct_complex(
    z_base: complex,
    shap_complex: np.ndarray,
    feature_names: list[str] | pd.Index,
    top_n: int = 8,
    show_polar_grid: bool = True,
    ax: plt.Axes | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """
    Visualiza a trajetória completa (caminhada vetorial) de predição
    no plano complexo (Argand-Gauss) para uma única amostra.
    """
    if len(shap_complex.shape) > 1:
        raise ValueError("sct_complex requer um array 1D (uma única amostra).")

    owns_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # --- 1. CÁLCULO DA CAMINHADA VETORIAL COMPLEXA ---
    magnitudes = np.abs(shap_complex)
    indices_ordenados = np.argsort(magnitudes)[::-1]

    caminhos_z = [z_base]
    labels_passos = ["Valor Base"]
    cores_vetores = []
    epsilon = 1e-5

    def cor_por_angulo_complexo(p: complex, s: complex):
        """Calcula o ângulo de fase relativo entre a posição atual e o vetor SHAP."""
        norm_p = np.abs(p)
        norm_s = np.abs(s)
        if norm_p < epsilon or norm_s < epsilon:
            sin_theta = 0.0
        else:
            # Produto vetorial 2D equivale a componente imaginária de s * conjugado(p)
            cross = p.real * s.imag - p.imag * s.real
            sin_theta = cross / (norm_p * norm_s)
            sin_theta = float(np.clip(sin_theta, -1.0, 1.0))

        peso = abs(sin_theta)
        cor_base = COR_ANTI_HORARIO if sin_theta > 0 else COR_HORARIO
        return _blend_color(cor_base, COR_RADIAL, peso)

    for idx in indices_ordenados[:top_n]:
        p = caminhos_z[-1]
        s = shap_complex[idx]
        cores_vetores.append(cor_por_angulo_complexo(p, s))
        caminhos_z.append(p + s)
        labels_passos.append(feature_names[idx])

    if top_n < len(shap_complex):
        p = caminhos_z[-1]
        s = np.sum(shap_complex[indices_ordenados[top_n:]])
        cores_vetores.append(cor_por_angulo_complexo(p, s))
        caminhos_z.append(p + s)
        labels_passos.append("Outras Features")

    z_pred = caminhos_z[-1]

    # --- 2. CONFIGURAÇÃO DA GRELHA E EIXOS ---
    ax.axhline(0, color="black", lw=1, alpha=0.3)
    ax.axvline(0, color="black", lw=1, alpha=0.3)

    max_mag = max(np.abs(z_base), np.abs(z_pred)) * 1.3
    if show_polar_grid:
        for r in np.linspace(0, max_mag, 6)[1:]:
            ax.add_patch(plt.Circle((0, 0), r, color="gray", ls="--", fill=False, alpha=0.2))

    # --- 3. DESENHO DA TRAJETÓRIA ---
    for i in range(len(caminhos_z) - 1):
        cor_atual = cores_vetores[i]
        z_atual, z_prox = caminhos_z[i], caminhos_z[i + 1]
        ax.annotate(
            "", xy=(z_prox.real, z_prox.imag), xytext=(z_atual.real, z_atual.imag),
            arrowprops=dict(arrowstyle="->", color=cor_atual, lw=2.5, shrinkA=0, shrinkB=0, zorder=4),
        )

    # Marcadores Base e Final
    ax.plot(z_base.real, z_base.imag, "s", color="black", markersize=8, label="Valor Base", zorder=6)
    ax.plot(z_pred.real, z_pred.imag, "*", color="black", markersize=14, label="Predição Final", zorder=6)

    # --- 4. PAINEL ANALÍTICO GERAL ---
    mag_base, fase_base = np.abs(z_base), np.angle(z_base, deg=True)
    mag_pred, fase_pred = np.abs(z_pred), np.angle(z_pred, deg=True)
    delta_mag = mag_pred - mag_base
    delta_fase = fase_pred - fase_base
    if delta_fase > 180: delta_fase -= 360
    elif delta_fase < -180: delta_fase += 360

    analytics_text = (
        rf"RESULTADO GLOBAL:" "\n"
        rf"$\Delta$ Mag: {delta_mag:+.2f}" "\n"
        rf"$\Delta$ Fase: {delta_fase:+.1f}°"
    )
    ax.text(0.05, 0.95, analytics_text, transform=ax.transAxes, ha="left", va="top",
            fontsize=10, bbox=dict(boxstyle="square,pad=0.5", fc="#f8f9fa", ec="gray", alpha=0.9), zorder=10)

    # --- 5. LEGENDAS E COLORBAR DE FASE ---
    ax.set_xlim(-max_mag, max_mag)
    ax.set_ylim(-max_mag, max_mag)
    ax.set_aspect("equal")
    ax.set_xlabel("Parte Real (Amplitude / Em Fase)")
    ax.set_ylabel("Parte Imaginária (Seno / Quadratura)")
    ax.set_title("Trajetória SHAP no Plano Complexo")
    
    elementos_legenda = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="black", markersize=8, label="Fasor Base"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="black", markersize=12, label="Fasor Predito"),
    ]
    ax.legend(handles=elementos_legenda, loc="lower right", fontsize="small")
    ax.grid(True, alpha=0.15)

    cmap_fase = LinearSegmentedColormap.from_list("fase_cmap", [COR_HORARIO, COR_RADIAL, COR_ANTI_HORARIO])
    cax = inset_axes(
        ax, width="32%", height="4%", loc="lower left",
        bbox_to_anchor=(0.02, 0.02, 1, 1), bbox_transform=ax.transAxes, borderpad=0,
    )
    gradiente = np.linspace(-1, 1, 256).reshape(1, -1)
    cax.imshow(gradiente, aspect="auto", cmap=cmap_fase, extent=[-1, 1, 0, 1])
    cax.set_yticks([])
    cax.set_xticks([-1, 0, 1])
    cax.set_xticklabels(["Horário", "Radial", "Anti-horário"], fontsize=6)
    cax.set_title("Sentido de Fase", fontsize=7, pad=2)
    for spine in cax.spines.values():
        spine.set_linewidth(0.5)

    plt.tight_layout()
    return _finalize(ax, owns_fig, show, save_path)


def bar_complex_mag(
    shap_complex: np.ndarray,
    feature_names: list[str] | pd.Index,
    top_n: int = 20,
    ax: plt.Axes | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """
    Calcula a magnitude global para arrays SHAP complexos (|Z|) 
    e repassa para o gráfico de barras padrão de magnitude.
    """
    shap_real = np.real(shap_complex)
    shap_imag = np.imag(shap_complex)
    
    # Reutiliza toda a lógica de plotagem global do módulo circular!
    return bar_mag(
        shap_x=shap_real,
        shap_y=shap_imag,
        feature_names=feature_names,
        top_n=top_n,
        ax=ax,
        show=show,
        save_path=save_path
    )

__all__ = ["scs_complex", "sct_complex", "bar_complex_mag"]