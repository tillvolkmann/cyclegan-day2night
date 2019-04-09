import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, get_filepath
from PIL import Image
import random
import json


class NightdriveDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires json file in nightdrive format with names of training images from domain A
    and from domain B, respectively. Nightdrive format is identical to BDD format, but a
    field "domain" must exist, indicating domain A or B.
    You can train the model with the dataset flag '--dataroot /path/to/data', all images
    must be located in this directory or subdirectories thereof.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument(
            '--jsonfile', type=str, default="", help='stem of json file containing names of images to be used; the stem is augmented by "_[opt.phase].json')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        # load json
        path_to_json = opt.jsonfile + ".json"  #  + "_" + opt.phase
        with open(path_to_json) as f:
            all_paths = json.load(f)
        # separate image names for domains A (night) and B (daytime)
        self.A_paths = [x['name'] for x in all_paths if x['domain'] == "A"]
        self.B_paths = [x['name'] for x in all_paths if x['domain'] == "B"]
        # get full paths to images in sub-directories
        for i in range(len(self.A_paths)):
            self.A_paths[i] = get_filepath(opt.dataroot, self.A_paths[i])
        for i in range(len(self.B_paths)):
            self.B_paths[i] = get_filepath(opt.dataroot, self.B_paths[i])
        # get the size of the data sets
        self.A_size = len(self.A_paths)  # get the size of data set A
        self.B_size = len(self.B_paths)  # get the size of data set B
        # get the number of channels of images
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image
        # apply image transforms
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
