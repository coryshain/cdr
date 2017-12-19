import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', rc={'axes.facecolor': (0, 0, 0, 0)})

def plot_convolutions(plot_x, plot_y, features, dir, filename='convolution_plot.jpg', irf_name_map=None, plot_x_inches=7, plot_y_inches=5, cmap='gist_earth'):
    cm = plt.get_cmap(cmap)
    plt.gca().set_prop_cycle(color=[cm(1. * i / len(features)) for i in range(len(features))])
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    feats = features[:]
    if irf_name_map is not None:
        for i in range(len(feats)):
            feats[i] = irf_name_map.get(feats[i], feats[i])
    sort_ix = [i[0] for i in sorted(enumerate(feats), key=lambda x:x[1])]
    for i in range(len(sort_ix)):
        if plot_y[1:,sort_ix[i]].sum() == 0:
            plt.plot(plot_x[:2], plot_y[:2,sort_ix[i]], label=feats[sort_ix[i]])
        else:
            plt.plot(plot_x, plot_y[:,sort_ix[i]], label=feats[sort_ix[i]])
    h, l = plt.gca().get_legend_handles_labels()
    plt.gcf().legend(h,l,fancybox=True, framealpha=0.5)
    plt.gcf().set_size_inches(plot_x_inches, plot_y_inches)
    plt.savefig(dir+'/'+filename)
    plt.close('all')

