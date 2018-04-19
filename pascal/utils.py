import numpy as np
import matplotlib.pyplot as plt


def palette(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def color_map_viz(p, labels, num_classes, height=50, width=500):
    array = np.empty(
        (height * (num_classes + 1), width, p.shape[1]), dtype=p.dtype)
    for i in range(num_classes):
        array[i*height:i*height+height, :] = p[i]
    array[num_classes*height:num_classes*height+height, :] = p[-1]

    plt.imshow(array)
    plt.yticks(
        [height * i + height / 2 for i in range(num_classes + 1)], labels)
    plt.xticks([])
    plt.show()


if __name__ == "__main__":
  labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                          'bus', 'car', 'cat', 'chair', 'cow',
                          'diningtable', 'dog', 'horse', 'motorbike', 'person',
                          'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
            'void']


  color_map_viz(palette(), labels, 21)
