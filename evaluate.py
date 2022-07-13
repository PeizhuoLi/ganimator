import numpy as np
import torch
import os
from os.path import join as pjoin
from demo import load_all_from_path, gen_noise
from models.architecture import draw_example, get_pyramid_lengths
from option import TestOptionParser
from evaluations.patched_nn import patched_nn_main
from evaluations.perwindow_nn import perwindow_nn, coverage
from tqdm import tqdm


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    test_parser = TestOptionParser()
    test_args = test_parser.parse_args()

    args, multiple_data, gens, z_stars, amps, lengths = load_all_from_path(test_args.save_path, test_args.device)
    device = torch.device(args.device)
    n_total_levels = len(gens)

    motion_data = multiple_data[0]

    noise_channel = z_stars[0].shape[1] if args.full_noise else 1

    print('levels:', lengths)
    save_path = pjoin(args.save_path, 'bvh')
    os.makedirs(save_path, exist_ok=True)

    base_img = motion_data.sample(size=lengths[0][0], slerp=args.slerp).to(device)

    motion_data.write(pjoin(save_path, 'base.bvh'), base_img)

    base_id = 0

    # Evaluate with reconstruct noise
    conds_rec = None
    for i in range(len(multiple_data)):
        motion_data = multiple_data[i]
        imgs = draw_example(gens, 'rec', z_stars[i], lengths[i] + [1], amps[i], 1, args, all_img=True, conds=conds_rec,
                            full_noise=args.full_noise)
        real = motion_data.sample(size=len(motion_data), slerp=args.slerp).to(device)
        motion_data.write(pjoin(save_path, f'gt_{i}.bvh'), real)
        motion_data.write(pjoin(save_path, f'rec_{i}.bvh'), imgs[-1])

        if imgs[-1].shape[-1] == real.shape[-1]:
            rec_loss = torch.nn.MSELoss()(imgs[-1], real).detach().cpu().numpy()
            print(f'Reconstruction loss: {rec_loss.item() * 1e5:.02f}')   # Scaling for better readability

    target_len = 2 * sum([len(data) for data in multiple_data])
    target_length = get_pyramid_lengths(args, target_len)
    while len(target_length) > n_total_levels:
        target_length = target_length[1:]
    z_length = target_length[0]

    amps2 = amps[base_id].clone()
    amps2[1:] = 0

    n_samples = 200

    print('Sampling...')
    all_samples = []
    loop = tqdm(range(n_samples))
    for i in loop:
        z_target = gen_noise(noise_channel, z_length, args.full_noise, device)
        z_target *= amps[base_id][0]
        imgs = draw_example(gens, 'random', z_stars[base_id], target_length, amps2, 1, args, all_img=True,
                            conds=None, full_noise=args.full_noise, given_noise=[z_target])
        all_samples.append(imgs[-1])
    loop.close()

    all_samples = torch.cat(all_samples, dim=0)
    all_samples = all_samples.permute(0, 2, 1)[..., :-6].detach().cpu()

    global_variations = []
    local_variations = []
    coverages = []

    for i in range(len(multiple_data)):
        motion_data = multiple_data[i]
        gt = motion_data.sample(size=len(motion_data), slerp=args.slerp).to(device)[0]
        gt = gt.permute(1, 0)[..., :-6].cpu()

        if len(motion_data) > 1:
            print(f'Evaluating on sequence {i}...')
        else:
            print('Evaluating...')

        loop = tqdm(range(n_samples))
        for i in loop:
            global_variations.append(patched_nn_main(all_samples[i], gt))
            local_variations.append(perwindow_nn(all_samples[i], gt, tmin=15))
            coverages.append(coverage(all_samples[i], gt))
        loop.close()

        print(f'Coverage: {np.mean(coverages) * 100:.1f}%')
        print(f'Global diversity: {np.mean(global_variations):.2f}')
        print(f'Local diversity: {np.mean(local_variations):.2f}')


if __name__ == '__main__':
    main()
