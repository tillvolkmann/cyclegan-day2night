from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        # Basic testing configuration
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')  # from code review, this has no function whatsover; see instead num_test below
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        # Added testing configuration
        # '--out_style' : "html" is original; "basic" outputs images directly in specified results folder and suffixes original filenames
        parser.add_argument('--out_style', type=str, default='html', help='output style of results ["html", "basic",  ]')
        parser.add_argument('--out_suffix', type=str, default='', help='output style of results')
        # Overwrite phase default
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')  # see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/285: Currently .eval() is not being used. I added it just in case others would like to use it. It should not affect CycleGAN model at all as CycleGAN has no dropout and batchnorm. It might affect pix2pix model. But we found that pix2pix can produce more diverse results when using dropout during the test time. You are free to try it for your own test code.
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run [-1 tests on all images in dataset]')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
