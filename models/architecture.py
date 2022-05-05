import os

import torch
from models.gan1d import Conv1dModel, GAN_model
from models.utils import get_interpolator
from os.path import join as pjoin
from tqdm import tqdm
import torch.nn.functional as F
from matplotlib import pyplot as plt
from functools import partial


def get_pyramid_lengths(args, dest):
    lengths = [16]
    ratio = eval(args.ratio)
    lengths[0] = int(dest * ratio)
    while lengths[-1] < dest:
        lengths.append(int(lengths[-1] * args.scaling_rate))
        if lengths[-1] == lengths[-2]:
            lengths[-1] += 1
    lengths[-1] = dest
    return lengths


def joint_train(all_reals, gens, gan_models, all_lengths, all_z_star, all_amps, args, loss_recorder,
                gt_deltas=None, ConGen=None):
    """
    Train several stages jointly
    :param reals: Training examples, size equal to current group size
    :param gens: All previously trained stages
    :param gan_models: Models to be trained
    :param lengths: Lengths including current group
    :param z_star: Reconstruction noise
    :param amps: Amplitude for nosie
    :param args: arguments
    :param loss_recorder: loss recorder
    """
    loop = range(args.num_iters)
    if not args.silent:
        loop = tqdm(loop)

    interpolator = get_interpolator(args)
    for iters in loop:
        for i in range(args.D_fact + args.G_fact + 1):
            for m in range(len(all_reals)):
                reals = all_reals[m]
                z_star = all_z_star[m]
                amps = all_amps[m]
                lengths = all_lengths[m]
                mode = 'random' if i < args.D_fact + args.G_fact else 'rec'
                if ConGen is not None and mode != 'rec':
                    conds = ConGen.random_generate(lengths)
                else:
                    conds = reals
                imgs = draw_example(gens, mode, z_star, lengths, amps, batch_size=1, args=args, all_img=True,
                                    full_noise=args.full_noise, conds=conds)
                imgs.append(torch.zeros_like(reals[0]))  # Trick for get the base image for stage_id = 0

                for j in range(len(gan_models)):
                    stage_id = j + len(gens) - len(gan_models)
                    gan_models[j].forward_proxy(reals[stage_id], interpolator(imgs[stage_id - 1], lengths[stage_id]))

                # Discriminator : Generator : Reconstruction = args.D_fact : args.G_fact : 1
                if i < args.D_fact:
                    optimize_lambda = lambda x: x.optimize_parameters(gen=False, disc=True, rec=False)
                elif i < args.D_fact + args.G_fact:
                    optimize_lambda = lambda x: x.optimize_parameters(gen=True, disc=False, rec=False)
                else:
                    optimize_lambda = lambda x: x.optimize_parameters(gen=False, disc=False, rec=True)
                list(map(optimize_lambda, gan_models))

        for j, gan_model in enumerate(gan_models):
            stage_id = j + len(gens) - len(gan_models)
            for loss_name in gan_model.loss_names:
                loss_recorder.add_scalar(f'{stage_id:02d}/{loss_name}', getattr(gan_model, loss_name))

            steps_save_path = pjoin(args.save_path, f'models/{stage_id:03d}')
            if args.save_freq != 0 and (iters + 1) % args.save_freq == 0:
                name = pjoin(steps_save_path, f'{(iters + 1) // args.save_freq:03d}x.pt')
                torch.save(gan_model.gen.state_dict(), name)


def realtime_example(gens, lengths, amps, conditions, trajs, args, layered_mask, full_noise=False):

    interpolater = get_interpolator(args)
    device = args.device

    imgs = []
    prev_img = torch.zeros((1, conditions[0].shape[1], lengths[0]), device=device)
    # prev_img[:, :conditions[0].shape[1]] = conditions[0]
    # prev_img[:, conditions[0].shape[1]:, layered_mask] = trajs
    for step, (gen, length, amp, cond, traj) in enumerate(zip(gens, lengths, amps, conditions, trajs)):
        length_cond = cond.shape[-1]
        length_traj = traj.shape[-1]
        n_channel = cond.shape[1] if full_noise else 1
        noise = torch.randn((1, n_channel, length), device=device) * amp
        if n_channel == 1:
            noise = noise.repeat(1, cond.shape[1], 1)
        noise = noise * torch.cat([
                            torch.zeros((1, length_cond), device=device),
                            torch.linspace(0, 1, length_traj).unsqueeze(0)
                        ], dim=-1)
        full_cond = torch.cat([cond[:, layered_mask], traj], dim=-1)
        if step < args.num_layered_generator:
            res = gen(noise + prev_img, prev_img, cond=full_cond, cond_requires_mask=False)
        else:
            res = gen(noise + prev_img, prev_img)

        res = res + prev_img
        if True:
            res[..., :length_cond] = cond[..., :length_cond]
        imgs.append(res)
        if step + 1 < len(lengths):
            prev_img = interpolater(res, size=lengths[step + 1])

    return imgs


def blend_vis(a, b, s, prefix=''):
    """
    Blending the time sequence a and b starting from time s
    :param a:
    :param b:
    :param s:
    :return:
    """
    from matplotlib import pyplot as plt
    delta = (a[..., :s] - b[..., :s]).squeeze()
    delta = (delta**2).mean(dim=0)
    plt.plot(delta.detach().cpu().numpy(), label=f'{prefix}_delta_{s}')
    pass


def blend(a, b, s):
    look_back = 10
    if s - look_back <= 0:
        return b
    res = b.clone()
    weight = torch.linspace(1, 0, look_back)
    res[..., s-look_back:s] = a[..., s-look_back:s] * weight + b[..., s-look_back:s] * (1 - weight)
    res[..., :s-look_back] = a[..., :s-look_back]
    return res


def sliding_window2(gens, z_star, amps, traj, args, step=30, look_back=20):
    noises = []
    imgs = []
    device = z_star.device

    lengths = get_pyramid_lengths(args, traj.shape[-1])

    for i in range(len(gens)):
        n_channel = z_star.shape[1] if args.full_noise else 1
        noise = torch.randn((1, n_channel, lengths[i]), device=device) * amps[i]
        noises.append(noise)

    start = 150
    total_lengths = []
    final_res = torch.zeros_like(traj)
    for i in range((lengths[-1] - start - 1) // step + 2):
        total_lengths.append(min(start + i * step, lengths[-1]))

    for i in range(len(total_lengths)):
        length_pyramid = get_pyramid_lengths(args, total_lengths[i])
        new_noise = [noises[i][..., :length_pyramid[i]] for i in range(len(length_pyramid))]
        new_traj = FullGenerator.downsample_generate(length_pyramid, traj[..., :total_lengths[i]])

        img = draw_example(gens, 'cond', z_star, length_pyramid, amps, 1, args, all_img=True,
                           conds=new_traj, full_noise=args.full_noise, given_noise=new_noise)

        imgs.append(img[-1])
        start_pt = 0 if i == 0 else total_lengths[i - 1] - look_back
        end_pt = total_lengths[i]
        final_res = blend(final_res, img[-1], start_pt)

        if 0 < i < 3:
            # blend_vis(final_res, img[-1], start_pt, '0')
            # blend_vis(imgs[-2], imgs[-1], start_pt, '1')
            pass
        # final_res[..., start_pt:end_pt] = imgs[-1][..., start_pt:]

    # plt.legend()
    # plt.show()
    return final_res, imgs


def build_data_pyramid(img, lengths, interpolator):
    res = []
    for length in lengths:
        res.append(interpolator(img, size=length))
    return res


def sliding_window(gens, amps, condition, traj, args, layered_mask, full_noise=False):
    linear_interpolator = partial(F.interpolate, mode='linear', align_corners=False)
    step = 50

    length_cond = condition.shape[-1]
    total_length = condition.shape[-1] + traj.shape[-1]

    all_lenghts = [length_cond + step]
    while all_lenghts[-1] < total_length:
        all_lenghts.append(min(all_lenghts[-1] + step, total_length))
    if all_lenghts[-1] - all_lenghts[-2] < 10:
        all_lenghts[-2] = all_lenghts[-1]
        all_lenghts = all_lenghts[:-1]

    last_cond = condition
    prev_length_pyramid = get_pyramid_lengths(args, length_cond)
    all_res = []
    all_res_origin = []
    for final_length in all_lenghts:
        length_pyramid = get_pyramid_lengths(args, final_length)
        tarj_pyramid = [length_pyramid[i] - prev_length_pyramid[i] for i in range(len(length_pyramid))]

        cur_trajs = build_data_pyramid(traj, tarj_pyramid, linear_interpolator)
        cur_conds = build_data_pyramid(last_cond, prev_length_pyramid, linear_interpolator)

        res = realtime_example(gens, length_pyramid, amps, cur_conds, cur_trajs, args, layered_mask, full_noise)
        cond = res[-1]
        all_res_origin.append(cond.clone())
        cond[..., :last_cond.shape[-1] - step//2] = last_cond[..., : -step//2]
        cond[:, layered_mask, length_cond:] = traj[..., :final_length - length_cond]
        prev_length_pyramid = length_pyramid
        last_cond = cond
        all_res.append(cond)

    return cond, all_res, all_res_origin


def draw_example(gens, mode, z_star, lengths, amps, batch_size, args, all_img=False, start_level=0,
                 conds=None, full_noise=False, num_cond=0, given_noise=None):
    """
    :param gens: list of generators at each stages
    :param mode: random, manip, cond, rec
    :param z_star: the noise corresponding to reconstruction at coarsest stage
    :param lengths: list of motion length at each stage
    :param amps: noise amplitude of each stage
    :param batch_size: batch_size
    :param args: args about motion
    :param all_img: requires all level images or not
    :param start_level: start level of non-reconstruction generation
    :param conds: base for editing(manip mode) or conditions for conditioned generation
    :param full_noise: if false, the noise will be shared by all the channel; else for "full" noise
    :param num_cond: number of stages of conditional generation
    :param given_noise: pre-defined noise for conditional generation
    :return: generated image(s)
    """
    if not isinstance(conds, list): conds = [conds]
    if args.layered_generator:
        num_cond = args.num_layered_generator
    else:
        # num_cond = 0
        pass
    device = args.device

    interpolater = get_interpolator(args)

    # Support interpolate z_star to different framerate to observe reconstruction result
    if z_star.shape[-1] != lengths[0]:
        z_star = interpolater(z_star, lengths[0])

    imgs = []
    if gens is None:
        return torch.zeros((batch_size, 1, lengths[0]), device=device)
    # with torch.no_grad():
    prev_img = torch.zeros((batch_size, z_star.shape[1], lengths[0]), device=device)
    for step, (gen, length, amp) in enumerate(zip(gens, lengths, amps)):
        current_mode = 'rec' if step < start_level else mode

        # Determine noise
        if args.no_noise:
            amp = 0.
        if current_mode == 'random' or current_mode == 'manip' or current_mode == 'cond':
            n_channel = z_star.shape[1] if full_noise else 1
            noise = torch.randn((batch_size, n_channel, length), device=device) * amp
            if given_noise is not None and step < len(given_noise):
                noise = given_noise[step]
            if noise.shape[1] == 1:
                noise = noise.repeat(1, z_star.shape[1], 1)
        elif current_mode == 'rec':
            noise = z_star if step == 0 else 0
        else:
            raise Exception('Unknown generating mode')

        # Execute generate model
        if current_mode == 'manip' and step < len(conds):
            prev_img = conds[step]
        elif step < num_cond:
            prev_img = gen(noise + prev_img, prev_img, cond=conds[step], cond_requires_mask=True) + prev_img
        else:
            prev_img = gen(noise + prev_img, prev_img) + prev_img

        # Save result and upsample for next level
        imgs.append(prev_img)
        if len(lengths) >= step + 2:
            prev_img = interpolater(prev_img, size=lengths[step + 1])

    return prev_img if not all_img else imgs


class FullGenerator:
    def __init__(self, args, motion_data, gens, z_star, amps):
        self.args = args
        self.motion_data = motion_data
        self.gens = gens
        self.z_star = z_star
        self.amps = amps

    def random_generate(self, lengths, down_sample=True):
        imgs_random = draw_example(self.gens[:len(lengths)], 'random', self.z_star, lengths + [1], self.amps, 1,
                                   self.args, all_img=True, start_level=0, conds=None,
                                   full_noise=self.args.full_noise)
        if down_sample:
            return self.downsample_generate(lengths, imgs_random[-1])
        else:
            return imgs_random

    @staticmethod
    def downsample_generate(lengths, input):
        res = []
        for length in lengths[:-1]:
            res.append(F.interpolate(input, size=length, mode='linear', align_corners=False))
        res.append(input)
        return res
