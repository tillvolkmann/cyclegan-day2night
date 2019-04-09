# import the necessary packages
import argparse
import cv2
import time
import os
import csv
from datetime import datetime
import numpy as np
import random

# https://www.pyimagesearch.com/2015/03/09/acapturing-mouse-click-events-with-python-and-opencv/


def mouse_events(event, x, y, flags, param):
    # grab references to the global variables
    global answer

    # check to see if the right mouse button was pressed, means real (0)
    if event == cv2.EVENT_LBUTTONDOWN:
        answer = 0
    # check to see if the right mouse button was pressed, means fake (1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        answer = 1


def read_image(im_path, im_width=None):
    # load the image
    im = cv2.imread(im_path)
    # resize (keeping aspect ratio)
    if im_width is not None:
        im_height = int(im_width / im.shape[1] * im.shape[0])
        im = cv2.resize(im, (im_width, im_height))
    return im


if __name__ == "__main__":
    """
        Example call:
        python3 ./scripts/eval_nightdrive/perceptual_turing_test.py \
          --folder_real /home/till/SharedFolder/CurrentDatasets/perceptual_studies/day2night_aws032_e14/real \
          --folder_fake /home/till/SharedFolder/CurrentDatasets/perceptual_studies/day2night_aws032_e14/fake \
          --folder_out /home/till/SharedFolder/CurrentDatasets/perceptual_studies/day2night_aws032_e14/output
          
    """

    # construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_real", required=True, help="Path to the image")
    parser.add_argument("--folder_fake", required=True, help="Path to the image")
    parser.add_argument("--folder_out", required=False, help="Path to the image")
    opt = parser.parse_args()

    # Settings
    timeout = 2.0
    num_test = 50
    num_warmup = 5
    wait_beforestart = 5
    answer_dict = {0: "real", 1: "fake", -999: "none given in time"}
    image_width = 640
    fake_str = "transfer"
    frac_fake = 0.5
    do_shuffle = True

    key_real = "q"
    key_fake = "p"

    # some helpers
    num_total = num_warmup + num_test
    num_fake_required = np.round(num_total * frac_fake)
    num_real_required = np.round(num_total * frac_fake)

    # Load image list
    list_images_real = os.listdir(opt.folder_real)
    list_images_fake = os.listdir(opt.folder_fake)
    num_real_avail = len(list_images_real)
    num_fake_avail = len(list_images_fake)
    num_min_avail = min([num_real_avail, num_fake_avail])
    if num_real_avail < num_real_required:
        raise Exception(f"Needed {num_real_required} real images, only found {num_real_avail} in folder {opt.folder_real}.")
    elif num_fake_avail < num_fake_required:
        raise Exception(f"Needed {num_fake_required} fake images, only found {num_fake_avail} in folder {opt.folder_fake}.")

    # shuffle
    if do_shuffle:
        random.shuffle(list_images_real)
        random.shuffle(list_images_fake)

    # initialize log variables
    test_answers = []
    test_images = []

    # User setup
    user_name = input("\n\n\nPlease enter your name (confirm by hitting ENTER): ")
    input(f"""\n

        ----------------------------------------------------------------------------------

        Thanks, {user_name}! Here are the instructions:

        You will be shown a total of {num_test} images that are either real or fake
        photos of night-time driving-related scenes.
        When presented with an image, hit {key_real} on the keyboard to identify
        the image as real, or {key_fake} to identify it as fake.

        You will be shown each image for exactly {np.round(timeout)} second(s) only. 
        You need to provide your answer (hit {key_real} or {key_fake} key) within that short
        time frame, so you need to decide very fast. 
        Once the next image appears, your input will be associated with that new
        image, so please do not try to provide a belated answer if you failed to 
        provide one in time.    

        The first {num_warmup} images that you will be shown are for warm-up, and 
        your selections for these images will not count. There will be no break after
        the warm-up sequence.

        >>> Remember, press {key_real} for real or {key_fake} for fake within one {np.round(timeout)} second(s) for each test image.

            Make sure you position your fingers on those keys before you start the test.

            Once you start the test, an empty window will first appear. After a 
            wait of {wait_beforestart} second(s) (a countdown is shown in the terminal),
            the test images will start appearing in that window.

            Once you are ready to start, please hit ENTER. <<<          

        ----------------------------------------------------------------------------------

        """)

    # initialize window with a blank image of correct size
    cv2.namedWindow("image")  # , cv2.WINDOW_AUTOSIZE ">>> hit 'r'eal or 'f'ake within one second <<<",
    cv2.setMouseCallback("image", mouse_events)
    image = read_image(os.path.join(opt.folder_real, list_images_real[0]), image_width)
    cv2.imshow("image", np.ones(image.shape)*1)
    cv2.waitKey(1) & 0xFF

    #
    print("Starting in ", sep='', end=': ', flush=True)
    for i in range(wait_beforestart, 0, -1):
        print(i, sep=' ... ', end=' ... ', flush=True)  # print in the same line by adding a "," at the end
        time.sleep(1)
        if i == 1:
            print("Let's start!")

    #
    random.seed(123)
    count_fake = 0
    count_real = 0
    for idx in range(num_test):

        # choose images for test
        rand_num = random.uniform(0, 1)
        if rand_num < frac_fake:
            image_path = os.path.join(opt.folder_fake, list_images_fake[count_fake])
            count_fake += 1
        else:
            image_path = os.path.join(opt.folder_real, list_images_real[count_real])
            count_real += 1

        # load the image and rescale it
        image = read_image(image_path, image_width)

        # keep looping until the 'q' key is pressed
        start_time = time.time()
        answer = -999  # "Not in time"
        while time.time() < start_time + timeout: #  or answer != -999: # != -999:
            # display the image and wait for a keypress
            cv2.imshow("image", image)
            key = cv2.waitKey(1) & 0xFF

            # if the 'r' key is pressed, means "real", break from the loop
            if key == ord(key_real):
                answer = 0
                # break
            # if the 'f' key is pressed, means "fake", break from the loop
            elif key == ord(key_fake):
                answer = 1
                # break

        print(f"Your last answer: '{answer_dict[answer]}'")
        test_answers.append(answer)
        test_images.append(image_path)

    # write log: name of image, whether fake or real, and answer
    if not os.path.exists(opt.folder_out):
        os.makedirs(opt.folder_out)
    out_file = "perceptual_study_" + user_name + datetime.now().strftime("%Y%m%dT%H%M%S") + ".csv"
    out_path = os.path.join(opt.folder_out, out_file)
    with open(out_path, mode='w') as csvFile:
        writer = csv.writer(csvFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["image_path", "is_warmup", "true_answer", "test_answer"])
        for idx in range(num_test):
            test_image = test_images[idx]
            if fake_str in test_image:
                true_answer = 1
            else:
                true_answer = 0
            if idx < num_warmup:
                is_warmup = 1
            else:
                is_warmup = 0
            writer.writerow([test_image, is_warmup, true_answer, test_answers[idx]])
        csvFile.close()

    # close all open windows
    cv2.destroyAllWindows()

