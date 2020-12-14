

def plot_data(X):
    """
    Generic function to plot the images in a grid
    of num_plot x num_plot
    :param X:
    :return:
    """
    plt.figure()
    num_plot = 5
    f, ax = plt.subplots(num_plot, num_plot)
    for i in range(num_plot):
        for j in range(num_plot):
            idx = np.random.randint(0, X.shape[0])
            ax[i,j].imshow(X[idx])
            ax[i,j].get_xaxis().set_visible(False)
            ax[i,j].get_yaxis().set_visible(False)
    f.subplots_adjust(hspace=0.1)  # No horizontal space between subplots
    f.subplots_adjust(wspace=0)
