from models.gan1d import GAN_model, Conv1dModel, LayeredGenerator, LayeredDiscriminator
import torch.nn as nn
from models.utils import get_layered_mask


def get_channels_list(args, dataset, neighbour_list):
    n_channels = dataset.n_channels

    joint_num = len(neighbour_list)

    base_channel = args.base_channel if args.base_channel != -1 else 128
    n_layers = args.n_layers if args.n_layers != -1 else 4
    if args.use_factor_channel_list:
        base_channel = n_channels

    channels_list = [n_channels]
    for i in range(n_layers - 1):
        channels_list.append(base_channel * (2 ** ((i+1) // 2)))
    channels_list += [n_channels]
    # channels_list = [n_channels, base_channel, 2*base_channel, 2*base_channel, n_channels]
    if args.skeleton_aware:
        channels_list = [((n - 1) // joint_num + 1) * joint_num for n in channels_list]
    if args.use_factor_channel_list:
        factor = [1, 1, 2, 2, 1]
        channels_list = [n_channels * f for f in factor]

    return channels_list


def get_group_list(args, num_stages):
    group_list = []
    for i in range(0, num_stages, args.group_size):
        group_list.append(list(range(i, min(i + args.group_size, num_stages))))
    return group_list


def create_layered_model(args, dataset, evaluation=False, channels_list=None):
    if args.new_layered:
        # In new implementation the layered model has been replaced by normal model.
        return create_model(args, dataset, evaluation, channels_list)
    n_channels = len(utils.get_layered_mask(args.layer_mode, dataset.n_rot))

    neighbour_list = dataset.bvh_file.get_neighbor(threshold=args.neighbour_dist, enforce_lower=args.enforce_lower,
                                                   enforce_contact=args.enforce_contact)

    channels_list_layered = [n_channels, n_channels, n_channels * 2, n_channels * 2, n_channels]
    channels_list_regular = get_channels_list(args, dataset, neighbour_list) if channels_list is None else channels_list

    if len(args.path_to_existing) != 0:
        layered_gen = None
    elif args.layered_full_receptive:
        layered_gen = Conv1dModel(channels_list_regular, args.kernel_size, last_active=None,
                                  padding_mode=args.padding_mode,
                                  batch_norm=args.batch_norm,
                                  neighbour_list=neighbour_list, skeleton_aware=args.skeleton_aware).to(args.device)
    else:
        layered_gen = Conv1dModel(channels_list_layered, args.kernel_size, last_active=None, padding_mode=args.padding_mode,
                                  batch_norm=args.batch_norm,
                                  neighbour_list=None, skeleton_aware=False).to(args.device)

    regular_gen = Conv1dModel(channels_list_regular, args.kernel_size, last_active=None, padding_mode=args.padding_mode,
                              batch_norm=args.batch_norm,
                              neighbour_list=neighbour_list, skeleton_aware=args.skeleton_aware).to(args.device)

    layered_gen = LayeredGenerator(args, layered_gen, regular_gen, dataset.n_rot, default_requires_mask=True)

    if evaluation:
        return layered_gen
    else:
        disc = Conv1dModel(channels_list_regular[:-1] + [1, ], args.kernel_size, last_active=None,
                           padding_mode=args.padding_mode, batch_norm=args.batch_norm,
                           neighbour_list=neighbour_list, skeleton_aware=args.skeleton_aware).to(args.device)

        if args.layered_discriminator:
            disc_layered = Conv1dModel(channels_list_layered[:-1] + [1, ], args.kernel_size, last_active=None,
                                       padding_mode=args.padding_mode, batch_norm=args.batch_norm,
                                       neighbour_list=None,
                                       skeleton_aware=False).to(args.device)

            disc = LayeredDiscriminator(args, disc_layered, disc, dataset.n_rot)

        gan_model = GAN_model(layered_gen, disc, args, dataset)
        return layered_gen, disc, gan_model


def create_model(args, dataset, evaluation=False, channels_list=None):
    if args.last_gen_active == 'None':
        gen_last_active = None
    elif args.last_gen_active == 'Tanh':
        gen_last_active = nn.Tanh()
    else:
        raise Exception('Unrecognized last_gen_active')

    neighbour_list = dataset.bvh_file.get_neighbor(threshold=args.neighbour_dist, enforce_contact=args.enforce_contact)
    if channels_list is None:
        channels_list = get_channels_list(args, dataset, neighbour_list)

    if not args.silent:
        print('Channel list:', channels_list)

    gen = Conv1dModel(channels_list, args.kernel_size, last_active=gen_last_active,
                      padding_mode=args.padding_mode, batch_norm=args.batch_norm,
                      neighbour_list=neighbour_list, skeleton_aware=args.skeleton_aware).to(args.device)
    if evaluation:
        return gen
    else:
        disc = Conv1dModel(channels_list[:-1] + [1,], args.kernel_size, last_active=None,
                           padding_mode=args.padding_mode, batch_norm=args.batch_norm,
                           neighbour_list=neighbour_list, skeleton_aware=args.skeleton_aware).to(args.device)
        gan_model = GAN_model(gen, disc, args, dataset)
        return gen, disc, gan_model
