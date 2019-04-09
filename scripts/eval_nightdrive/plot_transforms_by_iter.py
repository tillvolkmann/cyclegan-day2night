import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
import re


def plot_trainprogress(im_dir='', im_base=None, iters=None, max_num_img=10, domains=None, show=False, save_name=None,
                       max_n_iters=20):
    """
    Displays a selection of images for each
    Parameters:
        im_dir (str): path of directory of process images, e.g. '/home/till/projects/git-forks/pytorch-CycleGAN-and-pix2pix/results/cgan_aws_v021_by_iter_v0'
        im_base (list of str): base names of the original images, e.g. '7568f72d-520dedbb'
        iters (list of int): iters to consider, images with suffix "_iter_#" need to be available, e.g. [1,2,3]
        max_num_img (int): maximum number of images to be displayed (only relevant in case of folder based application)
    """

    # default values
    if domains is None:
        domains = ["A"]

    if save_name is None:
        save_name = os.path.basename(im_dir)

    # if iters are not provided, get all unique iters within im_dir
    if iters is None:
        list_img = list(os.listdir(im_dir))
        iters = []
        for im_name in list_img:
            if "iter_" in im_name:
                iter = re.findall(r'iter_([\d]*).jpg', im_name)
                if iter != []:  # this can happen due to word iter in postprocessing files
                    iters.append(int(*iter))
        iters = list(set(iters))

    # in any case, sort iters
    iters.sort()

    # if im_base is not provided, get all unique image base names within im_dir
    if im_base is None:
        list_img = list(os.listdir(im_dir))
        im_base = {k: [] for k in domains}  # defaultdict.fromkeys(domains, list().copy())
        for domain in domains:
            num_found = 0
            for im_name in list_img:
                if ("real_" + domain in im_name) and (
                        "iter_" + str(min(iters)) + "." in im_name):  # if this is an original image
                    im_s = im_name.split("_real")
                    im_base[domain].append(im_s[0])
                    num_found += 1
                if num_found == max_num_img:
                    break
            im_base[domain].sort()

    # create figure
    for domain in domains:

        iters_all = iters.copy()
        for i in range(0, len(iters_all), max_n_iters):
            print(i)
            iters = iters_all[i: i + max_n_iters - 1]
            print(iters)

            nrow = len(iters) + 1
            ncol = len(im_base[domain])
            figh = nrow * 3
            figw = ncol * 3 * (16 / 9)
            fig, ax = plt.subplots(nrow, ncol, figsize=[figw, figh], gridspec_kw={'wspace': 0.03, 'hspace': 0.03},
                                   squeeze=True)
            for col in range(ncol):  # for each base image
                im = im_base[domain][col]
                for row in range(nrow):  # for each iter
                    if row == 0:  # plot original
                        im_suf = im + "_real_" + domain + "_iter_{}".format(iters[0]) + ".jpg"

                    else:  # plot iter output
                        iter = iters[row - 1]
                        if domain == "A":
                            im_suf = im + "_transfer_AtoB" + "_iter_{}".format(iter) + ".jpg"
                        elif domain == "B":
                            im_suf = im + "_transfer_BtoA" + "_iter_{}".format(iter) + ".jpg"

                    try:
                        img = mpimg.imread(os.path.join(im_dir, im_suf))
                    except:
                        print("Image {} does not exist (e.g. because run was aborted), skipping.".format(
                            os.path.join(im_dir, im_suf)))
                        continue
                    # ax[row,col].axis("off")
                    imgplot = ax[row, col].imshow(img, interpolation="none")
                    ax[row, col].set_xticks([])
                    ax[row, col].set_yticks([])
                    if col == 0 and row == 0:
                        ax[row, col].set_ylabel("original", fontweight='bold', fontsize=20)
                    elif col == 0:
                        ax[row, col].set_ylabel("iter %d" % iter, fontweight='bold', fontsize=20)
                    if row == 0:
                        ax[row, col].set_title(im, fontweight='bold', fontsize=20)
                    # trying to make frame disappear
                    # ax[row,col].xaxis.set_visible(False)
                    # ax[row,col].yaxis.set_visible(False)
                    # fig.axes[row,col].get_xaxis().set_visible(False)

            # show figure
            if show:
                fig.show()

            # save figure to disk
            if save_name is not None:
                save_name_cur = save_name + '_iters{}to{}_domain{}.png'.format(min(iters), max(iters), domain)
                print("Saving figure file {}".format(os.path.join(im_dir, save_name_cur)))
                plt.savefig(os.path.join(im_dir, save_name_cur), format='png', dpi=150, bbox_inches='tight')


if __name__ == "__main__":

    # Example call on Docker:
    #   python3 /home/SharedFolder/git/tillvolkmann/pytorch-CycleGAN-and-pix2pix/scripts/eval_nightdrive/plot_transforms_by_iter.py

    # Specify host of this run
    which_host = "docker"

    # execute for specified host
    if which_host == "till":
        im_dir = '/home/till/projects/git-forks/pytorch-CycleGAN-and-pix2pix/results/cgan_aws_v029_trainprogress'
        iters = list(range(1, 24))
        plot_trainprogress(im_dir, max_num_img=10, domains=["A", "B"], save_name="cgan_aws_v029_trainprogress",
                           max_n_iters=26)  # iters=iters,

    elif which_host == "docker":
        im_dir = '/home/till/projects/git-forks/pytorch-CycleGAN-and-pix2pix/results/cgan_aws_v032_backupbeforerestart'
        im_dir = '/home/SharedFolder/git/tillvolkmann/pytorch-CycleGAN-and-pix2pix/results/cgan_aws_v032_backupbeforerestartb_trainprogress'

        iters = list(range(1, 24))
        plot_trainprogress(im_dir, max_num_img=30, domains=["A", "B"], save_name="cgan_aws_v029_trainprogress",
                           max_n_iters=26)  # iters=iters,

    else:
        raise Exception("Host unknown.")
