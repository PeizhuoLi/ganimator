import sys

def main():
    # Raise error if no path is provided
    if len(sys.argv) < 2:
        raise ValueError('Please provide a path to the BVH file.')
    bvh_path = sys.argv[1]
    joint_names = get_joints(bvh_path)
    print(joint_names)


def get_joints(bvh_path):
    with open('bvh_path', 'r') as f:
        lines = f.readlines()

    joint_names = []

    for line in lines:
        new_line = line.strip()
        if new_line.startswith('JOINT'):
            joint_name = new_line.split(' ')[1]
            joint_names.append(joint_name)

    return joint_names


if __name__ == '__main__':
    main()
