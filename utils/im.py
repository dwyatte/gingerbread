import numpy as np


def montage(images, n_rows=None, n_cols=None, scale=False, bg_color=0):
    """
    make a  montage of images
    :param images: 4d tensor (N, W, H, C)
    :param n_rows: number of rows
    :param n_cols: number of cols
    :param scale: scale each image to [0, 1]
    :param bg_color: background color of the montage
    :return:
    """
    n = images.shape[0]
    height = images.shape[1]
    width = images.shape[2]
    channels = images.shape[3]

    if n_rows and n_cols is None:
        n_cols = int(np.ceil(n/float(n_rows)))
    elif n_cols and n_rows is None:
        n_rows = int(np.ceil(n / float(n_cols)))
    elif n_rows is None and n_cols is None:
        n_rows = int(np.ceil(np.sqrt(n)))
        n_cols = int(np.ceil(np.sqrt(n)))

    canvas = np.ones(((n_rows*height)+n_rows+1, (n_cols*width)+n_cols+1, channels)) * bg_color

    image_count = 0
    row_off = 1

    for row in range(n_rows):
        col_off = 1
        for col in range(n_cols):
            if image_count == n:
                break

            image = images[image_count, :, :, :].copy()
            if scale:
                image -= image.min()
                image /= image.max()

            canvas[row_off:row_off+height, col_off:col_off+width, :] = image
            col_off += width+1
            image_count += 1

        row_off += height+1

    return canvas
