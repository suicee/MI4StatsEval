import matplotlib.pyplot as plt
import numpy as np



def plot_compare_bar(values, labels, colors, show_text=True, kw_bar=None, kw_text=None, kw_xticks=None):
    if kw_bar is None:
        kw_bar = {}
    if kw_xticks is None:
        kw_xticks = {}
    xx = np.arange(len(values))
    plt.bar(xx, values, color=colors, **kw_bar)
    if show_text:
        for x in xx:
            plt.text(x, values[x], '%.2f' % values[x], ha='center', **kw_text)
    plt.xticks(xx, labels, **kw_xticks)