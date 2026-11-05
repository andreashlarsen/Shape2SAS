from .sesans_calculations import calc_G_sesans, simulate_sesans

__all__ = [
    "calc_G_sesans",
    "simulate_sesans",
    "plot_sesans",
    "save_sesans",
]


def __getattr__(name):
    """import the output functions on first use, since they need matplotlib,
    which is too heavy to load whenever a structure factor is imported"""
    if name in ("plot_sesans", "save_sesans"):
        from . import sesans_output
        return getattr(sesans_output, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
