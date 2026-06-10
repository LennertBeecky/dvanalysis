"""DVAnalysis — Dynamic Retinal Vessel Analysis library.

Physiology-informed denoising and biomarker extraction for DVA signals.
"""

__version__ = "0.1.0"

try:
    from dvanalysis.pipeline import Pipeline
    __all__ = ["Pipeline", "__version__"]
except ImportError:
    __all__ = ["__version__"]
