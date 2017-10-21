import matplotlib.pylab as plt


def show_rmse():
    '''
    show figure
    '''
    # filmtrust
    # pmf=[0.85119,0.85401,0.86488,0.87981,0.90030] #PMF
    # socialReg=[0.83730,0.82967,0.84495,0.85632,0.87138] #SocialReg
    # SocialEmbeddings=[0.83607,0.82838,0.84288,0.85245,0.86950] #SocialEmbeddings
    # x=[0.8,0.7,0.6,0.5,0.4]

    # all users
    # mfr=[0.85119,0.85401,0.86488,0.87981,0.90030] #PMF
    # sr=[0.84357,0.84953,0.85599,0.86846,0.88905] #SocialReg
    # my=[0.83569, 0.84,0.84877,0.85367,0.87057] #SocialEmbeddings
    # x=[0.8,0.7,0.6,0.5,0.4]


    # cold start users
    # mfr=[0.922747,0.930411,0.965428,0.939825,0.950210]
    # sr=[0.917304,0.935492,0.954345,0.929423,0.928810]
    # my=[0.888466,0.91, 0.923517,0.903522,0.896970]
    # x=[0.8,0.7,0.6,0.5,0.4]

    # epinions
    mfr = [1.16677, 1.18048, 1.19479]
    sr = [1.09047, 1.10516, 1.13036]
    my = [1.08060, 1.09336, 1.11335]
    x = [0.8, 0.6, 0.4]

    plt.plot(x, mfr, label='PMF')
    plt.plot(x, sr, label='SocialReg')
    plt.plot(x, my, linewidth='2', label='SocialEmbeddings')

    plt.xlabel('ratio of train set')
    plt.ylabel('RMSE')
    plt.title('Epinions(d=30)-all users')
    plt.legend()
    plt.show()
    pass


show_rmse()


def plot_para(para, title, y_label):
    """
    show figure for rmse and epoch
    :param para:
    :param title:
    :param y_label:
    :return:
    """
    nums = range(len(para))
    plt.plot(nums, para, label=y_label)
    plt.xlabel('# of epoch')
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.savefig("../result/%s.png" % title)
    plt.close()
    pass
