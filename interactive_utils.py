import torch
from models.architecture import draw_example, get_pyramid_lengths, FullGenerator


def blend(a, b, s, look_back):
    """
    Blend the
    Parameters
    ----------
    0              s - look_back                s
    |              |                            |
        motion a
                     interp. between a and b
                                                    b
    Returns
    -------

    """
    if s - look_back <= 0:
        return b
    res = b.clone()
    weight = torch.linspace(1, 0, look_back)
    res[..., s-look_back:s] = a[..., s-look_back:s] * weight + b[..., s-look_back:s] * (1 - weight)
    res[..., :s-look_back] = a[..., :s-look_back]
    return res


def sliding_window(gens, z_star, amps, traj, args, step=30, look_back=30):
    noises = []
    imgs = []
    device = z_star.device

    lengths = get_pyramid_lengths(args, traj.shape[-1])

    for i in range(len(gens)):
        n_channel = z_star.shape[1] if args.full_noise else 1
        noise = torch.randn((1, n_channel, lengths[i]), device=device) * amps[i]
        noises.append(noise)

    start = 60                     # number of frames for warm-up
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
        start_pt = 0 if i == 0 else total_lengths[i - 1]
        final_res = blend(final_res, img[-1], start_pt, look_back)

    return final_res, imgs
