import matplotlib.pylab as plt


def show_rmse():
    '''
    show figure
    '''

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
