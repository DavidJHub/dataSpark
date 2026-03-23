"""
Visualization Themes
====================
Consistent color palettes and styling across all DataSpark charts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class Theme:
    """Centralized theme for all DataSpark visualizations."""

    # Palette
    primary: str = "#2563EB"
    secondary: str = "#7C3AED"
    success: str = "#059669"
    warning: str = "#D97706"
    danger: str = "#DC2626"
    neutral: str = "#6B7280"

    categorical_palette: list[str] = field(default_factory=lambda: [
        "#2563EB", "#7C3AED", "#059669", "#D97706", "#DC2626",
        "#0891B2", "#DB2777", "#4F46E5", "#65A30D", "#EA580C",
    ])

    diverging_cmap: str = "coolwarm"
    sequential_cmap: str = "Blues"
    figsize: tuple[int, int] = (10, 6)
    figsize_wide: tuple[int, int] = (14, 6)
    figsize_tall: tuple[int, int] = (10, 10)
    dpi: int = 100
    font_scale: float = 1.0
    style: str = "whitegrid"
    title_size: int = 14
    label_size: int = 11
    tick_size: int = 9
    alpha: float = 0.75

    def apply(self) -> None:
        """Apply theme globally to Matplotlib/Seaborn."""
        sns.set_style(self.style)
        sns.set_context("notebook", font_scale=self.font_scale)
        plt.rcParams.update({
            "figure.figsize": self.figsize,
            "figure.dpi": self.dpi,
            "axes.titlesize": self.title_size,
            "axes.labelsize": self.label_size,
            "xtick.labelsize": self.tick_size,
            "ytick.labelsize": self.tick_size,
        })

    @classmethod
    def dark(cls) -> "Theme":
        return cls(
            style="darkgrid",
            primary="#60A5FA",
            secondary="#A78BFA",
            success="#34D399",
            warning="#FBBF24",
            danger="#F87171",
            neutral="#9CA3AF",
            sequential_cmap="viridis",
        )

    @classmethod
    def minimal(cls) -> "Theme":
        return cls(style="ticks", font_scale=0.9)
