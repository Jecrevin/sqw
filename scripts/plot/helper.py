"""Helper functions for plotting."""

from matplotlib import pyplot as plt


def plt_style_setup() -> None:
    """Set up matplotlib plotting style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.labelsize": 16,
            "legend.fontsize": 12,
        }
    )
