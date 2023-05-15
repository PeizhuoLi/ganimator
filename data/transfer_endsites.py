import sys
import re


def main():
    # Raise error if no path is provided
    if len(sys.argv) < 3:
        raise ValueError('Please provide paths to the source and target BVH files.')
    source_file = sys.argv[1]
    target_file = sys.argv[2]
    fixed_file = target_file.replace('.bvh', '_offset_fixed.bvh')
    transfer_endsite_offsets(source_file, target_file, fixed_file)


def transfer_endsite_offsets(bvh1_path, bvh2_path, fixed_path):
    with open(bvh1_path, 'r') as f:
        lines = f.readlines()
    # Get the endsite offsets from the first BVH file
    endsite_offsets = []
    for i, line in enumerate(lines):
        line = line.strip()
        if line == 'End Site':
            offset = [float(x) for x in lines[i+2].strip().split()[1:]]
            endsite_offsets.append(offset)

    # Replace the endsite offsets in the second BVH file
    # with the ones from the first BVH file
    with open(bvh2_path, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if line == 'End Site':
            lines[i+2] = 'OFFSET {:.6f} {:.6f} {:.6f}\n'.format(*endsite_offsets.pop(0))
    with open(fixed_path, 'w') as f:
        f.writelines(lines) 


if __name__ == '__main__':
    main()
