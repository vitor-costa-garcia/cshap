"""
Visualizações SHAP para alvos circulares (ex.: ângulos, horários, direções)
codificados como componentes seno/cosseno.

Todas as funções públicas seguem o mesmo contrato:
- Recebem os arrays SHAP e, quando aplicável, um Axes opcional (`ax`).
- Se `ax` não for passado, a função cria sua própria Figure/Axes.
- Parâmetros comuns têm sempre o mesmo nome e a mesma posição relativa:
  ..., ax=None, show=True, save_path=None.
- Retornam sempre o `plt.Axes` utilizado, para permitir composição/edição
  posterior pelo chamador.
- Se `show=False`, a função nunca fecha a figura (o chamador ainda pode
  acessar/editar `ax.figure`). Se `show=True` e a função criou a própria
  figura, ela é fechada após a exibição.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .utils import (
    _add_quadrant_labels,
    _finalize,
    _blend_color,
    DEFAULT_CMAP,
    COR_ANTI_HORARIO,
    COR_HORARIO,
    COR_RADIAL,
    _QUADRANT_POSITIONS
)

# ---------------------------------------------------------------------------
# Funções públicas
# ---------------------------------------------------------------------------

def scs(
    shap_x: np.ndarray,
    shap_y: np.ndarray,
    feature_values: np.ndarray | pd.Series,
    feature_name: str = "Feature",
    cmap: str | LinearSegmentedColormap = DEFAULT_CMAP,
    quadrant_labels: dict[str, str] | None = None,
    ax: plt.Axes | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """Gera o gráfico de dispersão SHAP 2D para um alvo circular com suporte a labels."""
    owns_fig = ax is None  # rastreia se fomos nós que criamos a figura
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 8))

    # Eixos de referência central (0,0)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)

    # Gráfico de dispersão aplicando a cor
    scatter = ax.scatter(
        shap_x,
        shap_y,
        c=feature_values,
        cmap=cmap,
        alpha=0.7,
        edgecolors="none",
        s=40,
        zorder=3,
    )

    # Limites simétricos para manter a origem no centro exato
    max_val = max(np.abs(shap_x).max(), np.abs(shap_y).max()) * 1.15
    max_val = max_val if max_val > 0 else 1.0  # guard contra todos-zero
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    _add_quadrant_labels(ax, quadrant_labels)

    ax.set_xlabel("Impacto na Componente X (Cosseno)")
    ax.set_ylabel("Impacto na Componente Y (Seno)")
    ax.set_title(f"Dispersão Circular SHAP 2D: {feature_name}")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(f"Valor Real de {feature_name} (Azul=Baixo, Vermelho=Alto)")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    return _finalize(ax, owns_fig, show, save_path)


def sct(
    base_x: float,
    base_y: float,
    shap_x: np.ndarray,
    shap_y: np.ndarray,
    feature_names: list[str] | pd.Index,
    top_n: int = 8,
    quadrant_labels: dict[str, str] | None = None,
    confidence_radius: float | None = None,
    ax: plt.Axes | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """Gera um gráfico de Trajetória de Decisão SHAP 2D com Zona de Confiança e Análise Rotacional."""

    if len(shap_x.shape) > 1 or len(shap_y.shape) > 1:
        raise ValueError("sct requer arrays 1D (uma única amostra).")

    owns_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # --- 1. ZONA DE CONFIANÇA ---
    if confidence_radius:
        confidence_circle = plt.Circle(
            (base_x, base_y), confidence_radius,
            color="skyblue", alpha=0.15, label=f"Zona de Normalidade (R={confidence_radius})",
            zorder=1,
        )
        confidence_edge = plt.Circle(
            (base_x, base_y), confidence_radius,
            color="skyblue", fill=False, linestyle="--", linewidth=1.5, alpha=0.5,
            zorder=2,
        )
        ax.add_patch(confidence_circle)
        ax.add_patch(confidence_edge)

    # --- 2. CÁLCULO DA CAMINHADA VETORIAL E ROTAÇÃO ---
    magnitudes = np.sqrt(shap_x**2 + shap_y**2)
    indices_ordenados = np.argsort(magnitudes)[::-1]

    caminho_x, caminho_y = [base_x], [base_y]
    labels_passos = ["Valor Base"]
    cores_vetores = []

    epsilon = 1e-5

    def cor_por_angulo(px, py, sx, sy):
        """Calcula sin(theta) normalizado entre o vetor posição e o vetor passo,
        e retorna a cor interpolada entre COR_RADIAL e a cor de sentido correspondente."""
        norm_p = np.hypot(px, py)
        norm_s = np.hypot(sx, sy)

        if norm_p < epsilon or norm_s < epsilon:
            sin_theta = 0.0
        else:
            sin_theta = (px * sy - py * sx) / (norm_p * norm_s)
            sin_theta = float(np.clip(sin_theta, -1.0, 1.0))

        peso = abs(sin_theta)  # 0 = puramente radial, 1 = puramente tangencial
        cor_base = COR_ANTI_HORARIO if sin_theta > 0 else COR_HORARIO
        return _blend_color(cor_base, COR_RADIAL, peso)

    for idx in indices_ordenados[:top_n]:
        px, py = caminho_x[-1], caminho_y[-1]
        sx, sy = shap_x[idx], shap_y[idx]

        cores_vetores.append(cor_por_angulo(px, py, sx, sy))

        caminho_x.append(px + sx)
        caminho_y.append(py + sy)
        labels_passos.append(feature_names[idx])

    # Agrupamento "Outras Features"
    if top_n < len(shap_x):
        px, py = caminho_x[-1], caminho_y[-1]
        sx = np.sum(shap_x[indices_ordenados[top_n:]])
        sy = np.sum(shap_y[indices_ordenados[top_n:]])

        cores_vetores.append(cor_por_angulo(px, py, sx, sy))

        caminho_x.append(px + sx)
        caminho_y.append(py + sy)
        labels_passos.append("Outras Features")

    # --- 3. DESENHO DOS VETORES ---
    for i in range(len(caminho_x) - 1):
        cor_atual = cores_vetores[i]
        ax.annotate(
            "", xy=(caminho_x[i + 1], caminho_y[i + 1]), xytext=(caminho_x[i], caminho_y[i]),
            arrowprops=dict(arrowstyle="->", color=cor_atual, lw=2.5, shrinkA=0, shrinkB=0, zorder=4),
        )
        mx, my = (caminho_x[i] + caminho_x[i + 1]) / 2, (caminho_y[i] + caminho_y[i + 1]) / 2
        # ax.text(mx, my, labels_passos[i+1], fontsize=8, color=cor_atual, fontweight='bold',
        #         ha='center', va='center', bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8),
        #         zorder=5)

    # --- 4. MARCADORES, EIXOS E LIMITES ---
    ax.scatter(base_x, base_y, color="black", s=100, marker="s", label="Valor Base", zorder=6)
    ax.scatter(caminho_x[-1], caminho_y[-1], color="black", s=180, marker="*", label="Predição Final", zorder=6)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="-", alpha=0.3)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="-", alpha=0.3)

    all_points_x = caminho_x + [base_x + (confidence_radius or 0), base_x - (confidence_radius or 0)]
    all_points_y = caminho_y + [base_y + (confidence_radius or 0), base_y - (confidence_radius or 0)]

    max_dist_x = max(abs(min(all_points_x)), abs(max(all_points_x)))
    max_dist_y = max(abs(min(all_points_y)), abs(max(all_points_y)))
    max_val = max(max_dist_x, max_dist_y) * 1.15

    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    # --- 5. QUADRANTES E LEGENDA ---
    _add_quadrant_labels(ax, quadrant_labels)

    elementos_legenda = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="black", markersize=8, label="Valor Base"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="black", markersize=12, label="Predição Final"),
    ]

    if confidence_radius:
        elementos_legenda.insert(2, Patch(facecolor="skyblue", edgecolor="skyblue", alpha=0.3, linestyle="--",
                                           label=f"Normalidade (R={confidence_radius})"))

    ax.set_xlabel("Impacto na Componente X (Cosseno)")
    ax.set_ylabel("Impacto na Componente Y (Seno)")
    ax.set_title("Trajetória Circular SHAP (Dinâmica de Fase)")
    ax.legend(handles=elementos_legenda, loc="lower right", fontsize="small")
    ax.grid(True, alpha=0.1)

    # --- 6. COLORBAR DE GRADIENTE (ESPECTRO DE FASE) ---
    cmap_fase = LinearSegmentedColormap.from_list(
        "fase_cmap", [COR_HORARIO, COR_RADIAL, COR_ANTI_HORARIO]
    )
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


def bar_mag(
    shap_x: np.ndarray,
    shap_y: np.ndarray,
    feature_names: list[str] | pd.Index,
    top_n: int = 20,
    ax: plt.Axes | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """
    Gera um gráfico de barras global de Importância SHAP para modelos com alvo circular.
    Calcula a magnitude vetorial (Componentes X=Cosseno e Y=Seno) de TODAS as
    variáveis simultaneamente.
    """
    owns_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # 1. Calcula a matriz de magnitude para todas as predições e todas as features
    magnitude_matrix = np.sqrt(shap_x**2 + shap_y**2)

    # 2. Tira a média ao longo das linhas (axis=0) para obter a importância global de cada feature
    importances = pd.Series(magnitude_matrix.mean(axis=0), index=feature_names)

    # 3. Ordena e pega as top_n features para não poluir o gráfico
    importances_sorted = importances.sort_values(ascending=True).tail(top_n)

    # 4. Plotagem estilo SHAP
    ax.barh(importances_sorted.index, importances_sorted.values, color="#ff0051", edgecolor="none")

    ax.set_xlabel("Média da Magnitude SHAP (|Impacto Global Resultante|)")
    ax.set_title("Importância Global das Variáveis no Alvo Circular", pad=15)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # Estética limpa
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    return _finalize(ax, owns_fig, show, save_path)


__all__ = ["scs", "sct", "bar_mag"]