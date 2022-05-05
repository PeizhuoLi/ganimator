from bvh.bvh_parser import BVH_file
from os.path import join as pjoin
import numpy as np
import torch
from models.contact import constrain_from_contact
from models.kinematics import InverseKinematicsJoint2
from models.transforms import repr6d2quat
from tqdm import tqdm
import argparse


def continuous_filter(contact, length=2):
    contact = contact.copy()
    for j in range(contact.shape[1]):
        c = contact[:, j]
        t_len = 0
        prev = c[0]
        for i in range(contact.shape[0]):
            if prev == c[i]:
                t_len += 1
            else:
                if t_len <= length:
                    c[i - t_len:i] = c[i]
                t_len = 1
                prev = c[i]
    return contact


def fix_negative_height(contact, constrain, cid):
    floor = -1
    constrain = constrain.clone()
    for i in range(constrain.shape[0]):
        for j in range(constrain.shape[1]):
            if constrain[i, j, 1] < floor:
                constrain[i, j, 1] = floor
    return constrain


def fix_contact(bvh_file, contact):
    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    cid = bvh_file.skeleton.contact_id
    glb = bvh_file.joint_position()
    rotation = bvh_file.get_rotation(repr='repr6d').to(device)
    position = bvh_file.get_position().to(device)
    contact = contact > 0.5
    # contact = continuous_filter(contact)
    constrain = constrain_from_contact(contact, glb, cid)
    constrain = fix_negative_height(contact, constrain, cid).to(device)
    cid = list(range(glb.shape[1]))
    ik_solver = InverseKinematicsJoint2(rotation, position, bvh_file.skeleton.offsets.to(device), bvh_file.skeleton.parent,
                                        constrain[:, cid], cid, 0.1, 0.01, use_velo=True)

    loop = tqdm(range(500))
    for i in loop:
        loss = ik_solver.step()
        loop.set_description(f'loss = {loss:.07f}')

    return repr6d2quat(ik_solver.rotations.detach()), ik_solver.get_position()


def fix_contact_on_file(prefix, name):
    try:
        contact = np.load(pjoin(prefix, name + '.bvh.contact.npy'))
    except:
        print(f'{name} not found')
        return
    bvh_file = BVH_file(pjoin(prefix, name + '.bvh'), no_scale=True, requires_contact=True)
    print('Fixing foot contact with IK...')
    res = fix_contact(bvh_file, contact)
    bvh_file.writer.write(pjoin(prefix, name + '_fixed.bvh'), res[0], res[1], names=bvh_file.skeleton.names, repr='quat')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    args = parser.parse_args()
    if args.prefix[0] == '/':
        prefix = args.prefix
    else:
        prefix = f'./results/{args.prefix}'
    name = args.name
    contact = np.load(pjoin(prefix, name + '.bvh.contact.npy'))
    bvh_file = BVH_file(pjoin(prefix, name + '.bvh'), no_scale=True, requires_contact=True)

    res = fix_contact(bvh_file, contact)

    bvh_file.writer.write(pjoin(prefix, name + '_fixed.bvh'), res[0], res[1], names=bvh_file.skeleton.names, repr='quat')
