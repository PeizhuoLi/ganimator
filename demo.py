import torch
import os
from os.path import join as pjoin
from dataset.motion import MotionData, load_multiple_dataset
from models import create_model, create_layered_model
from models.architecture import draw_example, get_pyramid_lengths, FullGenerator
from option import TestOptionParser, TrainOptionParser
from fix_contact import fix_contact_on_file
from models.utils import get_layered_mask


def load_all_from_path(save_path, device, use_class=False):
    train_parser = TrainOptionParser()
    args = train_parser.load(pjoin(save_path, 'args.txt'))
    args.device = device
    args.save_path = save_path
    device = torch.device(args.device)

    if not args.multiple_sequences:
        motion_data = MotionData(pjoin(args.bvh_prefix, f'{args.bvh_name}.bvh'),
                                 padding=args.skeleton_aware, use_velo=args.use_velo, repr=args.repr,
                                 contact=args.contact, keep_y_pos=args.keep_y_pos)
        multiple_data = [motion_data]
    else:
        multiple_data = load_multiple_dataset(prefix=args.bvh_prefix, name_list=pjoin(args.bvh_prefix, args.bvh_name),
                                              padding=args.skeleton_aware, use_velo=args.use_velo, repr=args.repr,
                                              contact=args.contact, keep_y_pos=args.keep_y_pos,
                                              no_scale=True)
        motion_data = multiple_data[0]

    lengths = []
    min_len = 10000
    for i in range(len(multiple_data)):
        new_length = get_pyramid_lengths(args, len(multiple_data[i]))
        min_len = min(min_len, len(new_length))
        if args.num_stages_limit != -1:
            new_length = new_length[:args.num_stages_limit]
        lengths.append(new_length)

    for i in range(len(multiple_data)):
        lengths[i] = lengths[i][-min_len:]

    gens = []
    for step, length in enumerate(lengths[0]):
        create = create_layered_model if args.layered_generator and step < args.num_layered_generator else create_model
        gen = create(args, motion_data, evaluation=True)
        try:
            gen_sate = torch.load(pjoin(args.save_path, f'gen{step:03d}.pt'), map_location=device)
        except FileNotFoundError:
            gen_sate = torch.load(pjoin(args.save_path, f'gen{step}.pt'), map_location=device)
        gen.load_state_dict(gen_sate)
        gens.append(gen)
    z_star = torch.load(pjoin(args.save_path, 'z_star.pt'), map_location=device)
    amps = torch.load(pjoin(args.save_path, 'amps.pt'), map_location=device)
    if use_class:
        if isinstance(z_star, list):
            z_star = z_star[0]
        if len(amps.shape) != 1:
            amps = amps[0]
        return FullGenerator(args, motion_data, gens, z_star, amps)
    else:
        if len(amps.shape) == 1:
            amps = amps.unsqueeze(0)
        if isinstance(z_star, torch.Tensor) and len(z_star.shape) == 3:
            z_star = z_star.unsqueeze(0)
        return args, multiple_data, gens, z_star, amps, lengths


def write_multires(imgs, prefix, writer, interpolator, full_lengths=None, requires_con_loss=True):
    os.makedirs(prefix, exist_ok=True)
    length = imgs[-1].shape[-1] if full_lengths is None else full_lengths
    res = []
    for step, img in enumerate(imgs):
        full_length = interpolator(img, length)
        writer(pjoin(prefix, f'{step:02d}.bvh'), full_length)
        velo = full_length[:, -6:-3].norm(dim=1)
        res.append(velo)
    if requires_con_loss:
        res = torch.cat(res, dim=0)
        consistency_loss = torch.nn.MSELoss()(res[1], res[0])
        return consistency_loss


def gen_noise(n_channel, length, full_noise, device):
    if full_noise:
        res = torch.randn((1, n_channel, length)).to(device)
    else:
        res = torch.randn((1, 1, length)).repeat(1, n_channel, 1).to(device)
    return res


def main():
    test_parser = TestOptionParser()
    test_args = test_parser.parse_args()

    args, multiple_data, gens, z_stars, amps, lengths = load_all_from_path(test_args.save_path, test_args.device)
    device = torch.device(args.device)
    n_total_levels = len(gens)

    motion_data = multiple_data[0]

    noise_channel = z_stars[0].shape[1] if args.full_noise else 1

    if len(args.path_to_existing):
        ConGen = load_all_from_path(args.path_to_existing, args.device, use_class=True)
        ConGen.output_mask = get_layered_mask(args.layer_mode, motion_data.n_rot)
        conds_rec = [motion_data.sample(lengths[0][i]) for i in range(args.num_layered_generator)]
    else:
        ConGen = None
        conds_rec = None

    print('levels:', lengths)
    save_path = pjoin(args.save_path, 'bvh')
    os.makedirs(save_path, exist_ok=True)

    base_id = 0

    # Evaluate with reconstruct noise
    for i in range(len(multiple_data)):
        motion_data = multiple_data[i]
        imgs = draw_example(gens, 'rec', z_stars[i], lengths[i] + [1], amps[i], 1, args, all_img=True, conds=conds_rec,
                            full_noise=args.full_noise)
        real = motion_data.sample(size=len(motion_data), slerp=args.slerp).to(device)
        motion_data.write(pjoin(save_path, f'gt_{i}.bvh'), real)
        motion_data.write(pjoin(save_path, f'rec_{i}.bvh'), imgs[-1])

        if imgs[-1].shape[-1] == real.shape[-1]:
            rec_loss = torch.nn.MSELoss()(imgs[-1], real).detach().cpu().numpy()
            print(f'rec_loss: {rec_loss.item():.07f}')

    generation_mode = 'manip' if test_args.style_transfer or test_args.keyframe_editing else 'random'

    if test_args.style_transfer:
        manip_data = MotionData(f'{test_args.style_transfer}',
                                padding=args.skeleton_aware, use_velo=args.use_velo, repr=args.repr,
                                contact=args.contact, keep_y_pos=args.keep_y_pos,
                                no_scale=False)
        target_len = len(manip_data)
        target_length = get_pyramid_lengths(args, target_len)
        conds = manip_data.sample(target_length[0], mode='nearest')
    elif test_args.keyframe_editing:
        manip_data = MotionData(f'{test_args.keyframe_editing}',
                                padding=args.skeleton_aware, use_velo=args.use_velo, repr=args.repr,
                                contact=args.contact, keep_y_pos=args.keep_y_pos,
                                no_scale=True)
        target_len = len(multiple_data[0])                       # Use original length of training data
        target_length = get_pyramid_lengths(args, target_len)
        conds = manip_data.sample(target_length[0])
    elif test_args.conditional_generation:
        manip_data = MotionData(f'{test_args.conditional_generation}',
                                padding=args.skeleton_aware, use_velo=args.use_velo, repr=args.repr,
                                contact=args.contact, keep_y_pos=args.keep_y_pos,
                                no_scale=True)
        target_len = len(manip_data)  # Use original length of training data
        target_length = get_pyramid_lengths(args, target_len)
        conds = [manip_data.sample(l) for l in target_length[:args.num_layered_generator]]
        generation_mode = 'cond'
    elif args.layered_generator:
        "This is a conditional model, but the condition is not given. Then the condition will be sampled from the ConGen model"
        target_len = test_args.target_length
        target_length = get_pyramid_lengths(args, target_len)
        manip_data = None
        conds = ConGen.random_generate(target_length)
        conds_full = conds[-1]
        conds = conds[:args.num_layered_generator]
    else:
        target_len = test_args.target_length
        target_length = get_pyramid_lengths(args, target_len)
        manip_data = None
        conds = None

    while len(target_length) > n_total_levels:
        target_length = target_length[1:]
    z_length = target_length[0]

    z_target = gen_noise(noise_channel, z_length, args.full_noise, device)
    z_target *= amps[base_id][0]

    amps2 = amps[base_id].clone()
    amps2[1:] = 0

    imgs = draw_example(gens, generation_mode, z_stars[base_id], target_length, amps2, 1, args, all_img=True,
                        conds=conds, full_noise=args.full_noise, given_noise=[z_target])
    motion_data.write(pjoin(save_path, f'result.bvh'), imgs[-1])
    fix_contact_on_file(save_path, name=f'result')

    if manip_data is not None:
        motion_data.write(pjoin(save_path, 'manipulate_input.bvh'), manip_data.sample())
    elif args.layered_generator:
        motion_data.write(pjoin(save_path, 'generated_traj.bvh'), conds_full)


if __name__ == '__main__':
    main()
