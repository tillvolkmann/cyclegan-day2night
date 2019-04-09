from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html [unit iter]')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console [unit iter]')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters (see also base_options "additional parameters" for continue train options)
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results [unit iter]')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs [unit epoch]')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration (@tv: this means that, rather than saving a "latest" model at --save_latest_freq and overwriting it, a model is saved with suffix "iter" and stored; this is in addition to saving by epoch and, as said, contoled by save_latest_freq')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...  (@tv contols where in the scheduler we are upon continue train)' )
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--niter', type=int, default=100, help='number of iter at starting learning rate (@tv NO, this is actually number for epochs!)')
        parser.add_argument('--niter_decay', type=int, default=100, help='number of iter to linearly decay learning rate to zero (@tv: NO, this is number of epochs. So default is to keep the LR const for 100 epochs, then decay to zero at 100+100 epochs')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper. Wasserstein GAN is implemented according to this paper https://arxiv.org/abs/1704.00028, see this issue https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/439.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations (this only applies if lr_policy is step)')

        self.isTrain = True
        return parser
