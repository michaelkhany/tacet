"""Figures and reports. Requires the optional ``viz`` extra for plotting."""

from .cloud_plot import plot_cloud, plot_components
from .report import render_report, write_report

__all__ = ["plot_cloud", "plot_components", "render_report", "write_report"]
