"""
cycshap - SHAP Explainability for Circular Data

This module provides tools for interpreting machine learning models
trained on circular/cyclic data (angles, directions, phases, periodic phenomena).
"""

from . import utils
from .circular import scs, sct, bar_mag
from .complex import scs_complex, sct_complex, bar_complex_mag

__version__ = "0.1.0"
__author__ = "cycshap contributors"
__license__ = "MIT"

# Explicitly define public API
__all__ = [
    # Circular functions
    "scs",
    "sct",
    "bar_mag",
    # Complex functions
    "scs_complex",
    "sct_complex",
    "bar_complex_mag",
    # Submodules
]

# Make functions easily accessible
__doc__ = """
cycshap: SHAP Explainability for Circular Data
==============================================

Modules:
- circular: Core functions for circular SHAP (scs, trajectory_plot, bar_cycmag)
- complex: SHAP on the complex plane (scs_complex, plot_complex_shap_step)
- utils: Utility functions (blend_color, geometric operations)

Quick Start:
    >>> import cycshap
    >>> explainer = cycshap.scs(model, X_circular)
    >>> shap_values = explainer.shap_values(X_test)
    >>> cycshap.trajectory_plot(shap_values, X_test)
"""
