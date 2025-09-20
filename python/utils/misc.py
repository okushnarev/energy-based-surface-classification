import numpy as np
import matplotlib
from matplotlib import pyplot as plt


def init_matplotlib():
    matplotlib.rcParams['pdf.fonttype'] = 42
    plt.rcParams.update({
        'savefig.dpi':     800,
        'font.family':     'Serif',
        'font.serif':      'Times New Roman',
        'font.size':       16,
        'axes.axisbelow':  True,

        "axes.labelpad":   12,
        "axes.labelsize":  16,

        "xtick.bottom":    False,
        "ytick.left":      False,

        "xtick.major.pad": 10,
        "ytick.major.pad": 15,
    })


my_pal = {
    'gray':  '#b6b6b6',
    'green': '#4fc54c',
    'table': '#9f6a4d',
    'red': '#ad3024'
}

def hex_to_rgba(hex_color, alpha=0.3):
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})'


def cosine_func(x, B, A_1, A_2, scale=71 / 355):
    return A_1 * np.cos(np.pi * (x * scale * 4 / 72 + 2 / 72)) + A_2 * np.cos(
        np.pi * (x * scale * 12 / 72 + 6 / 72)) + B
