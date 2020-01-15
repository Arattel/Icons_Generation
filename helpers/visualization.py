from matplotlib import pyplot as plt


def visualize_images(arr, function=lambda x: x, cmap=None, figsize=(8, 8), rows=5, columns=4):
    fig = plt.figure(figsize=figsize)
    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(function(arr[i]), cmap=cmap)
    plt.show()
