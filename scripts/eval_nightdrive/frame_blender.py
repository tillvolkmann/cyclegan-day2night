from PIL import Image
import os
import re
import argparse


def blend_frame(frame1, frame2, frac_frame2=0.5, frac_transition=0.01):
    """
    Blends parts of two frames into one frame of the same size as the original frames.

    :param frame1: First input frame, will be on the left-hand side of the output frame.
    :param frame2: Second input frame, will be on the right-hand side of the output frame.
    :param frac_frame2: Width (horizontal) fraction of the output frame taken up by frame2.
    :param frac_transition: Width (horizontal) fraction of the image that forms a smooth (mixed) transition between the left and right part,
    :return: Blended frame


    """
    # creates a mask indicating the part where frame2 is inserted
    mask = Image.new('RGBA', frame1.size, color=(0, 0, 0, 255))
    width_rightpart = int(mask.size[0] * frac_frame2)
    height_rightpart = mask.size[1]
    rightpart = Image.new('RGBA', (width_rightpart, height_rightpart), color=(0, 0, 0, 0))
    start_rightpart = mask.size[0] - width_rightpart
    mask.paste(rightpart, (start_rightpart, 0))

    # create a transition zone in the mask between the two frames
    if frac_transition > 0:
        width_transition = int(mask.size[0] * frac_transition)
        height_transition = mask.size[1]
        start_transition = int(max(0, (start_rightpart - (width_transition / 2))))
        end_transition = int(min(mask.size[0], (start_rightpart + (width_transition / 2))))
        for i in range(width_transition):
            transition_part = Image.new('RGBA', (1, height_transition),
                                        color=(0, 0, 0, int(255 - i / width_transition * 255)))
            mask.paste(transition_part, (start_transition + i, 0))

    # blend the two frames and return
    return Image.composite(frame1, frame2, mask)


def horzcat_frames(*images, pad_fraction=0.01, pad_color=(0, 0, 0)):
    """
    Horizontally concatenate two or more images into a single image.

    :param images: Two or more images as returned by PIL.Image.open.
    :param pad_fraction: Height of the padding between frames, given as fraction of the total horizontal extent of the
    concatenated output image.
    :param pad_color: Color to use for the padding. Default is black. If given, this should be a single integer
    or floating point value for single-band modes, and a tuple for multi-band modes (one value
    per band). When creating RGB images, you can also use color strings as supported by the ImageColor
    module. If the color is None, the padding is transparent.
    For more info on color mode, see https://pillow.readthedocs.io/en/3.0.x/reference/ImageColor.html
    :return: Horizontally concatenated images as a single image.
    """

    # get dimensions
    total_width = 0
    max_height = 0
    for im in images:
        max_height = max(im.size[1], max_height)
        total_width += im.size[0]

    num_images = len(images)

    x_pad = int(round(total_width * pad_fraction))
    total_width += int((num_images - 1) * x_pad)

    im_concat = Image.new('RGB', (total_width, max_height), color=pad_color)

    x_offset = 0
    for im in images:
        im_concat.paste(im, (x_offset, 0))
        x_offset += im.size[0] + x_pad

    return im_concat


def vertcat_frames(*images, pad_fraction=0.01, pad_color=(0, 0, 0)):
    """
    Vertically concatenate two or more images into a single image.

    :param images: Two or more images as returned by PIL.Image.open.
    :param pad_fraction: Height of the padding between frames, given as fraction of the total vertical extent of the
    concatenated output image.
    :param pad_color: Color to use for the padding. Default is black. If given, this should be a single integer
    or floating point value for single-band modes, and a tuple for multi-band modes (one value
    per band). When creating RGB images, you can also use color strings as supported by the ImageColor
    module. If the color is None, the padding is transparent.
    For more info on color mode, see https://pillow.readthedocs.io/en/3.0.x/reference/ImageColor.html
    :return: Vertically concatenated images as a single image.
    """

    # get dimensions
    total_height = 0
    max_width = 0
    for im in images:
        max_width = max(im.size[0], max_width)
        total_height += im.size[1]

    num_images = len(images)
    y_pad = int(round(total_height * pad_fraction))
    total_height += int((num_images - 1) * y_pad)

    if pad_color is None:
        pad_color = (255, 255, 255, 0)

    im_vertcat = Image.new('RGB', (max_width, total_height), color=pad_color)

    y_offset = 0
    for c, im in enumerate(images):
        im_vertcat.paste(im, (0, y_offset))
        if c == 1:
            y_offset += y_pad
        else:
            y_offset += im.size[1] + y_pad

    return im_vertcat


if __name__ == "__main__":
    """
    Blend, horizontally concatenate, and vertically concatenate all image pairs in a folder.

    Example call:
    $ python3 ./scripts/eval_nightdrive/frame_blender.py --path ~/SharedFolder/CurrentDatasets/bdd100k_videos_selected/00a04f65-8c891f94 --suffix _transfer_AtoB --fpc 60
    $ python3 ./scripts/eval_nightdrive/frame_blender.py --path ~/SharedFolder/CurrentDatasets/bdd100k_test --suffix _transfer_AtoB --fpc 60
    """

    # Test settings
    # opt.dir = "/home/till/projects/git-forks/pytorch-CycleGAN-and-pix2pix/results/videos/YouTube"
    # opt.suffix = "_transfer_AtoB"
    # opt.fpc = 300  # frames_per_cycle

    # Instantiate parser
    parser = argparse.ArgumentParser()
    # Arguments
    parser.add_argument('--path', type=str, required=True, help='Path to directory containing frames.')
    parser.add_argument('--pattern1', type=str, required=False,
                        help='Pattern identifying first version of frame, e.g. "frame-" will use files names "frame-[0-9]*"')
    parser.add_argument('--pattern2', type=str, required=True,
                        help='Pattern identifying scond version of frame, e.g. "frame-" will use files names "frame-[0-9]*"')
    parser.add_argument('--fpc', type=int, default=300, help='frames per cycle.')
    parser.add_argument('--out_basename', type=str, required=False, help='Base name of output frames, e.g. "frame-" will give "frame-blended-[0-9]".')

    #
    parser.add_argument('--blend', action='store_true')
    parser.add_argument('--vertcat', action='store_true')
    parser.add_argument('--horzcat', action='store_true')
    parser.set_defaults(blend=False, vertcat=False, horzcat=False, pattern1="frame-", out_basename="frame-")

    # Get arguments
    opt = parser.parse_args()

    # get list of all frames of type 1 to process, i.e. those that contain "opt.suffix"
    search_pattern_1 = opt.pattern1 + r"\d\d*"
    list_frames_1 = [fn for fn in os.listdir(opt.path) if re.search(search_pattern_1, fn) is not None]
    # get list of all frames of type 1 to process, i.e. those that contain "opt.suffix"
    search_pattern_2 = opt.pattern2 + r"\d\d*"
    list_frames_2 = [fn for fn in os.listdir(opt.path) if re.search(search_pattern_2, fn) is not None]
    if len(list_frames_2) == 0:
        raise Exception(f"No frames containing {opt.pattern2} found in {opt.path}.")

    # get common frames
    frame_counter_1 = [int(*re.findall(r"\d\d*", f)) for f in list_frames_1]
    frame_counter_2 = [int(*re.findall(r"\d\d*", f)) for f in list_frames_2]
    frame_counter_1 = set(frame_counter_1)
    frame_counter = list(frame_counter_1.intersection(set(frame_counter_2)))
    list_frames_1 = [fn for fn in list_frames_1 if int(*re.findall(r"\d\d*", fn)) in frame_counter]
    list_frames_2 = [fn for fn in list_frames_2 if int(*re.findall(r"\d\d*", fn)) in frame_counter]

    # sort the frames by their counter
    # list 1
    frame_counter = [int(*re.findall(r"\d\d*", f)) for f in list_frames_1]
    zipped = zip(frame_counter, list_frames_1)
    zipped = sorted(zipped, key=lambda k: k[0])
    frame_counter, list_frames_1 = map(list, zip(*zipped))
    # list 2
    frame_counter = [int(*re.findall(r"\d\d*", f)) for f in list_frames_2]
    zipped = zip(frame_counter, list_frames_2)
    zipped = sorted(zipped, key=lambda k: k[0])
    frame_counter, list_frames_2 = map(list, zip(*zipped))

    # get the sequence of fractions of the right image shown for each frame
    fracs_rightpart = [min(i % opt.fpc, -i % opt.fpc) / (opt.fpc / 2) for i in range(len(list_frames_2))]

    # blend original and suffixe'ed frame
    num_frames = len(list_frames_1)
    for i, fn1 in enumerate(list_frames_1):

        # Import a pair of frames
        fn_base, fn_ext = os.path.splitext(fn1)
        frame1 = Image.open(os.path.join(opt.path, fn1))
        fn2 = list_frames_2[i]
        frame2 = Image.open(os.path.join(opt.path, fn2))

        # Blend frames by partially overlapping them
        if opt.blend:
            frame_blended = blend_frame(frame1, frame2, frac_frame2=fracs_rightpart[i], frac_transition=0.01)
            frame_blended.save(os.path.join(opt.path, opt.out_basename + 'blended-' + str(frame_counter[i]) + fn_ext))  # Save blended frame

        # Concat frames vertically
        if opt.vertcat:
            frame_vertcat = vertcat_frames(frame1, frame2, pad_fraction=0.0, pad_color=None)
            frame_vertcat.save(os.path.join(opt.path, opt.out_basename + 'vertcat-' + str(frame_counter[i]) + fn_ext))

        # Concat frames horizontally
        if opt.horzcat:
            frame_horzcat = horzcat_frames(frame1, frame2, pad_fraction=0.00, pad_color=None)
            frame_horzcat.save(os.path.join(opt.path, opt.out_basename + 'horzcat-' + str(frame_counter[i]) + fn_ext))

        # Print progress message
        if (i+1) % 100 == 0:
            print("Processed {} of {} frames.".format(i+1, num_frames))

