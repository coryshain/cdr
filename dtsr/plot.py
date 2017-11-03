import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

def plot_convolutions(plot_x, plot_y, features, dir, filename='convolution_plot.jpg', fixef_name_map=None, plot_x_inches=7, plot_y_inches=5, cmap='gist_earth'):
    plt.gcf().set_size_inches(plot_x_inches, plot_y_inches)
    cm = plt.get_cmap(cmap)
    plt.gca().set_prop_cycle(color=[cm(1. * i / len(features)) for i in range(len(features))])
    n_feat = plot_y.shape[-1]
    feats = features[:]
    if fixef_name_map is not None:
        for i in range(len(feats)):
            feats[i] = fixef_name_map[features[i]]
    for i in range(n_feat):
        plt.plot(plot_x, plot_y[:,i], label=feats[i])
    plt.legend()
    plt.savefig(dir+'/'+filename)
    plt.clf()

