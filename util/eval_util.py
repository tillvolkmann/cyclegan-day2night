import os
import re
import pandas as pd
from . import util
from scipy.misc import imresize


def parse_loss_log(name):
    """
    Parse the loss_log.txt file generated during training of cycle_gan model.

    Parameters:
        name : str
            name of the experiment / training run, or absolute path to a log_loss txt file

    Returns:
        df : pandas data frame
            log data of the experiment / training run

    """
    # check whether the name of an experiment was provided, or an absolute path to a file
    if os.path.isfile(name):
        path = name
    else:
        path = os.path.join("checkpoints/", name, "loss_log.txt")

    # parse the file
    data = []
    with open(path, "r") as f:
        f.readline()  # skip first line
        line = f.readline()
        names = re.findall(r"\b[a-zA-Z]\w*", line)  # get variable names
        while line:
            digits = map(float, re.findall(r"\d[\d.]*", line))  # get all numerical values
            data.append(digits)
            line = f.readline()

    # return as pandas data frame
    return pd.DataFrame(data, columns=names)


def save_images_basic(opt, data, results_path, visuals, aspect_ratio=1.0, width=1280):
    """Save images to the disk.
    Images written include:
    -

    Parameters:
        opt
        data
        results_path
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.

    in cycle_ga_model.set_input, AtoB = self.opt.direction == 'AtoB' is considered

    Some notes on incoming formats:
    image_path :
        a list containing only one of the paths, e.g. ['/home/till/data/driving/BerkeleyDeepDrive/bdd100k_sorted/test/b35a415a-02526f57.jpg']
    data :
        {'A': tensor([[[[ 0.0353,  0.0275,  0.0118,  ..., -0.1843, -0.0902, -0.2000],
           ...  [-0.7333, -0.7333, -0.7333,  ..., -0.6706, -0.6706, -0.6706]]]]),
        'B': tensor([[[[-1.0000, -1.0000, -1.0000,  ..., -0.9608, -0.9608, -0.9608],
          ...  [-0.9843, -0.9843, -0.9843,  ..., -0.9765, -0.9765, -0.9765]]]]),
        'A_paths': ['/home/till/data/driving/BerkeleyDeepDrive/bdd100k_sorted/test/b35a415a-02526f57.jpg'],
        'B_paths': ['/home/till/data/driving/BerkeleyDeepDrive/bdd100k_sorted/test/4feefa31-18e668bd.jpg']}
        ['/home/till/data/driving/BerkeleyDeepDrive/bdd100k_sorted/test/b35a415a-02526f57.jpg']
    for each pair (label, image_tensor) within an
        ordered dict, e.g. OrderedDict([('real_A', tensor([[[[ 0.0353,  0.0275,  0.0118,  ..., -0.1843, -0.0902, -0.2000], ...
    """

    if opt.dataset_mode == 'single':
        if opt.direction == 'AtoB':
            path_real_A = data['A_paths']  # list of paths
            name_real_A, ext_real_A = os.path.splitext(os.path.basename(*path_real_A))
        else:
            path_real_B = data['B_paths']  # list of paths
            name_real_B, ext_real_B = os.path.splitext(os.path.basename(*path_real_B))

    else:
        path_real_A = data['A_paths']  # list of paths
        path_real_B = data['B_paths']
        name_real_A, ext_real_A = os.path.splitext(os.path.basename(*path_real_A))
        name_real_B, ext_real_B = os.path.splitext(os.path.basename(*path_real_B))

    # for each image result (real, fake, identity, each A and B)
    for label, im_data in visuals.items():

        # get and process image
        im = util.tensor2im(im_data)
        h, w, _ = im.shape
        if aspect_ratio > 1.0:
            im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
        elif aspect_ratio < 1.0:
            im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')

        # construct output image name
        if any([x == opt.out_style for x in ["basic", "basic_single"]]):
            if 'real_A' in label:
                image_name = name_real_A + '_real_A' + opt.out_suffix + ext_real_A
            elif 'fake_B' in label:
                image_name = name_real_A + '_transfer_AtoB' + opt.out_suffix + ext_real_A
            elif 'rec_A' in label:
                image_name = name_real_A + '_rec_A' + opt.out_suffix + ext_real_A
            elif 'real_B' in label:
                image_name = name_real_B + '_real_B' + opt.out_suffix + ext_real_B
            elif 'fake_A' in label:
                image_name = name_real_B + '_transfer_BtoA' + opt.out_suffix + ext_real_B
            elif 'rec_B' in label:
                image_name = name_real_B + '_rec_B' + opt.out_suffix + ext_real_B
            else:
                continue

        elif any([x == opt.out_style for x in ["conversion", "conversion_single"]]):
            if 'fake_B' in label:
                image_name = name_real_A + '_transfer_AtoB' + opt.out_suffix + ext_real_A
            elif 'rec_A' in label:
                image_name = name_real_A + '_rec_A' + opt.out_suffix + ext_real_A
            elif 'fake_A' in label:
                image_name = name_real_B + '_transfer_BtoA' + opt.out_suffix + ext_real_B
            elif 'rec_B' in label:
                image_name = name_real_B + '_rec_B' + opt.out_suffix + ext_real_B
            else:
                continue

        elif any([x == opt.out_style for x in ["frames"]]):
            if 'fake_B' in label:
                image_name = re.sub(r"([0-9]+)", r"{}\1".format('transfer_AtoB-'), name_real_A) + opt.out_suffix + ext_real_A
            elif 'rec_A' in label:
                image_name = re.sub(r"([0-9]+)", r"{}\1".format('rec_A-'), name_real_A) + opt.out_suffix + ext_real_A
            elif 'fake_A' in label:
                image_name = re.sub(r"([0-9]+)", r"{}\1".format('transfer_BtoA-'), name_real_B) + opt.out_suffix + ext_real_B
            elif 'rec_B' in label:
                image_name = re.sub(r"([0-9]+)", r"{}\1".format('rec_B-'), name_real_B) + opt.out_suffix + ext_real_B
            else:
                continue

        # save image
        save_path = os.path.join(results_path, image_name)
        util.save_image(im, save_path)


def save_images_progress(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        h, w, _ = im.shape
        if aspect_ratio > 1.0:
            im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
        if aspect_ratio < 1.0:
            im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
        util.save_image(im, save_path)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


if __name__ == "__main__":
    pass
