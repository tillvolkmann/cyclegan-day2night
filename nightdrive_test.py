"""Test script for day-to-night image translation in project night-drive.

Once a model is trained with train_nightdrive.py, this script can be used to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    Options:
        Basic:
        '--model <model> ~ name of model (e.g. cyclegan)
            '--model test' ~ This option is used for generating CycleGAN results only for one side.
            This option will automatically set '--dataset_mode single', which only loads the images from one set.
            On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
            which is sometimes unnecessary.
        '--results_dir <directory_path_to_save_result>' Use to specify the writing directory for the results.
            By default, the results will be saved at ./results/, or ./results/name/ if name is given.
        '--name <name_of_the_experiment> ~ Name of the run / experiment (e.g. maps_cyclegan), i.e. name of the sub-directory in checkpoints/
        '--dataset_mode <dataset_mode> ~
        '--dataroot <path to data> ~ path to data (e.g. ./datasets/maps)
        Test a pix2pix model:
            python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
        '--preprocess
        Evaluation:
        '--eval' : use eval mode during test time.
        '--num_test <num_test (int)>' : how many test images to run (default=50)



See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md

CycleGAN is quite memory-intensive as four networks (two generators and two discriminators) need to be loaded on one
GPU, so a large image cannot be entirely loaded. In this case, we recommend training with cropped images. For
example, to generate 1024px results, you can train with --preprocess scale_width_and_crop --load_size 1024
--crop_size 360, and test with --preprocess scale_width --crop_size 1024. This way makes sure the training and test
will be at the same scale. At test time, you can afford higher resolution because you don’t need to load all
networks.

General settings for night-drive:
  python3 nightdrive_test.py
    --model cycle_gan --phase test --no_dropout
    --preprocess none --load_size 1280
    --gpu_ids -1
    --name cgan_aws_v021

To process our data sets, use:
  Very quick model evaluation
    --dataset_mode nightdrive
    --dataroot /home/till/data/driving/BerkeleyDeepDrive/bdd100k_sorted/
    --jsonfile /home/till/data/driving/BerkeleyDeepDrive/bdd100k_sorted/bdd100k_sorted_cgan_small
    --results_dir ./results/cgan_aws_v021d/
  Quick model evaluation
    --dataset_mode nightdrive
    --dataroot /home/till/data/driving/BerkeleyDeepDrive/bdd100k_sorted/
    --jsonfile /home/till/data/driving/BerkeleyDeepDrive/bdd100k_sorted/bdd100k_sorted_cgan
    --results_dir ./results/cgan_aws_v021/
  train_A:
    --dataset_mode deepdrive
    --dataroot /home/till/data/driving/BerkeleyDeepDrive/bdd100k_sorted/train_A/
    --jsonfile /home/till/data/driving/BerkeleyDeepDrive/bdd100k_sorted/train_A/bdd100k_sorted_train_A
    --results_dir /home/till/data/driving/BerkeleyDeepDrive/bdd100k_sorted/train_A_gan/
  test:
    --dataset_mode deepdrive
    --dataroot /home/till/data/driving/BerkeleyDeepDrive/bdd100k_sorted/test/
    --jsonfile /home/till/data/driving/BerkeleyDeepDrive/bdd100k_sorted/test/bdd100k_sorted_test
    --results_dir /home/till/data/driving/BerkeleyDeepDrive/bdd100k_sorted/test_gan/
  valid:
    --dataset_mode deepdrive
    --dataroot /home/till/data/driving/BerkeleyDeepDrive/bdd100k_sorted/valid/
    --jsonfile /home/till/data/driving/BerkeleyDeepDrive/bdd100k_sorted/valid/bdd100k_sorted_valid
    --results_dir /home/till/data/driving/BerkeleyDeepDrive/bdd100k_sorted/valid_gan/


  Regarding produced image resolution, I tested the following for our model established using --load_size 1280 --preprocess scale_width_and_crop --crop_size 360
  ~ nothing specified at all <> makes small test images
  --preprocess scale_width --crop_size 1280 <> makes 1280x720 images
  --preprocess none --load_size 1280 <> makes 1280x720 images, looks identical to previous

"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from util.eval_util import save_images_basic, save_images_progress


if __name__ == '__main__':

    # get test options, parsing user input
    opt = TestOptions().parse()
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # set number of test images to "all" if opt.num_test -1
    if opt.num_test == -1:
        opt.num_test = len(dataset)

    # create a website
    if opt.out_style == 'html':
        web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    elif any([x == opt.out_style for x in ['basic', 'basic_single', 'frames']]):
        out_dir = os.path.join(opt.results_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    elif any([x == opt.out_style for x in ['conversion', 'conversion_single']]):
        out_dir = os.path.join(opt.results_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    elif opt.out_style == 'progress':
        web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
        webpage = html.HTML(web_dir, 'Epoch eval --- Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    # test with eval mode. This only affects layers like batchnorm and dropout.
    #   For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    #   For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    # output results
    for i, data in enumerate(dataset):

        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break

        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths

        if i % 10 == 0:  # print progress
            print('processing (%04d)-th image... %s' % (i, img_path))

        if opt.out_style == 'html':  # save images to an HTML file
            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        elif any([opt.out_style == x for x in ['basic', 'basic_single', 'conversion', 'conversion_single', 'frames']]):
            save_images_basic(opt, data, out_dir, visuals, aspect_ratio=opt.aspect_ratio)
        elif opt.out_style == 'progress':
            save_images_progress(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    if any([x == opt.out_style for x in ['html', 'progress']]):
        webpage.save()  # save the HTML
