import argparse
import sys
import os


class OptionParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--debug', type=int, default=0)
        self.parser.add_argument('--device', type=str, default='cpu')
        self.parser.add_argument('--gan_mode', type=str, default='wgan-gp')
        self.parser.add_argument('--save_path', type=str, default='./results/')
        self.parser.add_argument('--padding_mode', type=str, default='reflect')
        self.parser.add_argument('--batch_norm', type=int, default=0)
        self.parser.add_argument('--scaling_rate', type=float, default=1/0.75)
        self.parser.add_argument('--kernel_size', type=int, default=5)
        self.parser.add_argument('--bvh_name', type=str, default='dance')
        self.parser.add_argument('--bvh_prefix', type=str, default='./data/')
        self.parser.add_argument('--last_gen_active', type=str, default='None')
        self.parser.add_argument('--skeleton_aware', type=int, default=1)
        self.parser.add_argument('--neighbour_dist', type=int, default=2)
        self.parser.add_argument('--use_velo', type=int, default=1)
        self.parser.add_argument('--ratio', type=str, default='1./5')
        self.parser.add_argument('--no_noise', type=int, default=0)
        self.parser.add_argument('--no_gan', type=int, default=0)
        self.parser.add_argument('--repr', type=str, default='repr6d')
        self.parser.add_argument('--activation', type=str, default='LeakyReLu')
        self.parser.add_argument('--contact', type=int, default=1)
        self.parser.add_argument('--enforce_contact', type=int, default=1)
        self.parser.add_argument('--slerp', type=int, default=0)
        self.parser.add_argument('--nearest_interpolation', type=int, default=0)
        self.parser.add_argument('--conditional_generator', type=int, default=0)
        self.parser.add_argument('--conditional_mode', type=str, default='locrot')
        self.parser.add_argument('--full_noise', type=int, default=0)
        self.parser.add_argument('--num_conditional_generator', type=int, default=1)
        self.parser.add_argument('--keep_y_pos', type=int, default=1)
        self.parser.add_argument('--path_to_existing', type=str, default='')
        self.parser.add_argument('--num_stages_limit', type=int, default=-1)
        self.parser.add_argument('--group_size', type=int, default=2)
        self.parser.add_argument('--multiple_sequences', type=int, default=0)
        self.parser.add_argument('--joint_reduction', type=int, default=1)
        self.parser.add_argument('--use_factor_channel_list', type=int, default=0)
        self.parser.add_argument('--base_channel', type=int, default=-1)
        self.parser.add_argument('--n_layers', type=int, default=-1)

    @staticmethod
    def checker(args):
        if args.slerp:
            raise Exception('Slerp is no longer supported.')
        if args.nearest_interpolation and args.conditional_generator:
            raise Exception('Conditional with nearest interpolation not yet implemented')
        if args.multiple_sequences and len(args.path_to_existing) > 0:
            raise Exception('Does not support conditional generation for multiple sequences.')
        if not args.contact and (args.enforce_contact or args.enforce_lower):
            raise Exception('Contact is not enabled, but enforce_contact or enforce_lower is enabled.')
        return args

    def parse_args(self, args_str=None):
        return self.checker(self.parser.parse_args(args_str))

    def get_parser(self):
        return self.parser

    def save(self, filename, args_str=None):
        if args_str is None:
            args_str = ' '.join(sys.argv[1:])
        path = '/'.join(filename.split('/')[:-1])
        os.makedirs(path, exist_ok=True)
        with open(filename, 'w') as file:
            file.write(args_str)

    def load(self, filename):
        with open(filename, 'r') as file:
            args_str = file.readline()
        return self.parse_args(args_str.split())


class TrainOptionParser(OptionParser):
    def __init__(self):
        super(TrainOptionParser, self).__init__()
        self.parser.add_argument('--D_fact', type=int, default=5)
        self.parser.add_argument('--G_fact', type=int, default=1)
        self.parser.add_argument('--lr_g', type=float, default=1e-4)
        self.parser.add_argument('--lr_d', type=float, default=1e-4)
        self.parser.add_argument('--shared_lr', type=float, default=1e-4)
        self.parser.add_argument('--num_iters', type=int, default=15000)
        self.parser.add_argument('--lambda_gp', type=float, default=1)
        self.parser.add_argument('--lambda_rec', type=float, default=50.)
        self.parser.add_argument('--silent', type=int, default=0)
        self.parser.add_argument('--rec_loss_type', type=str, default='L1')
        self.parser.add_argument('--lambda_consistency', type=float, default=5.)
        self.parser.add_argument('--detach_label', type=int, default=0)
        self.parser.add_argument('--use_sigmoid', type=int, default=1)
        self.parser.add_argument('--save_freq', type=int, default=0)
        self.parser.add_argument('--requires_noise_amp', type=int, default=1)
        self.parser.add_argument('--full_zstar', type=int, default=1)
        self.parser.add_argument('--correct_zstar_gen', type=int, default=0)
        self.parser.add_argument('--use_6d_fk', type=int, default=0)

    @staticmethod
    def checker(args):
        args = OptionParser.checker(args)
        if args.no_gan:
            args.D_fact = 0
        if args.shared_lr != 0:
            args.lr_g = args.shared_lr
            args.lr_d = args.shared_lr
        return args


class TestOptionParser(OptionParser):
    def __init__(self):
        super(TestOptionParser, self).__init__()
        self.parser.add_argument('--target_length', type=int, default=600)
        self.parser.add_argument('--style_transfer', type=str, default='')
        self.parser.add_argument('--keyframe_editing', type=str, default='')
        self.parser.add_argument('--conditional_generation', type=str, default='')
