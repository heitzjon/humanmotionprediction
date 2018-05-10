import numpy as np
import pandas as pd


class Skeleton(object):
    """
    Represents a skeleton, i.e. defines how the joints are linked together and the offset for each bone.
    """
    end_effectors = np.array([5, 10, 15, 21, 23, 29, 31])

    parents = np.array([0, 1, 2, 3, 4, 5, 1, 7, 8, 9, 10, 1, 12, 13, 14, 15, 13,
                        17, 18, 19, 20, 21, 20, 23, 13, 25, 26, 27, 28, 29, 28, 31]) - 1

    n_joints = len(parents)

    offsets = np.array([[0.000000, 0.000000, 0.000000],
                        [-132.948591, 0.000000, 0.000000],
                        [0.000000, -442.894612, 0.000000],
                        [0.000000, -454.206447, 0.000000],
                        [0.000000, 0.000000, 162.767078],
                        [0.000000, 0.000000, 74.999437],
                        [132.948826, 0.000000, 0.000000],
                        [0.000000, -442.894413, 0.000000],
                        [0.000000, -454.206590, 0.000000],
                        [0.000000, 0.000000, 162.767426],
                        [0.000000, 0.000000, 74.999948],
                        [0.000000, 0.100000, 0.000000],
                        [0.000000, 233.383263, 0.000000],
                        [0.000000, 257.077681, 0.000000],
                        [0.000000, 121.134938, 0.000000],
                        [0.000000, 115.002227, 0.000000],
                        [0.000000, 257.077681, 0.000000],
                        [0.000000, 151.034226, 0.000000],
                        [0.000000, 278.882773, 0.000000],
                        [0.000000, 251.733451, 0.000000],
                        [0.000000, 0.000000, 0.000000],
                        [0.000000, 0.000000, 99.999627],
                        [0.000000, 100.000188, 0.000000],
                        [0.000000, 0.000000, 0.000000],
                        [0.000000, 257.077681, 0.000000],
                        [0.000000, 151.031437, 0.000000],
                        [0.000000, 278.892924, 0.000000],
                        [0.000000, 251.728680, 0.000000],
                        [0.000000, 0.000000, 0.000000],
                        [0.000000, 0.000000, 99.999888],
                        [0.000000, 137.499922, 0.000000],
                        [0.000000, 0.000000, 0.000000]])


def exp2rotmat(expmap):
    """
    Converts joint angles in exponential map format to 3-by-3 rotation matrices. This is an implementation of the
    Rodrigues formula.
    :param expmap: np array of shape (N, 3)
    :return: np array of shape (N, 3, 3)
    """
    theta = np.linalg.norm(expmap, axis=-1, keepdims=True)
    axis = expmap / (theta + np.finfo(np.float32).eps)
    cost = np.cos(theta)[..., np.newaxis]
    sint = np.sin(theta)[..., np.newaxis]
    rrt = np.matmul(axis[..., np.newaxis], axis[:, np.newaxis, ...])
    skew = np.zeros([expmap.shape[0], 3, 3])
    skew[:, 0, 1] = -axis[:, 2]
    skew[:, 0, 2] = axis[:, 1]
    skew[:, 1, 2] = -axis[:, 0]
    skew = skew - np.transpose(skew, [0, 2, 1])
    R = np.repeat(np.eye(3, 3)[np.newaxis, ...], expmap.shape[0], axis=0)
    R = R*cost + (1 - cost)*rrt + sint*skew
    return R


def forward_kinematics(expmap, skeleton=Skeleton, root_pos=None):
    """
    Compute joint positions from joint angles represented as exponential maps.
    :param expmap: np array of shape (seq_length, n_joints*3)
    :param skeleton: skeleton defining parents and offsets per joint
    :param root_pos: root positions per frame as an np array of shape (seq_length, 3) or None
    :return: np array of shape (seq_length, n_joints, 3)
    """
    seq_length = expmap.shape[0]
    n_joints = expmap.shape[1] // 3

    roots = np.zeros([seq_length, 3]) if root_pos is None else root_pos
    angles = np.reshape(expmap, [seq_length, n_joints, 3])

    if n_joints < skeleton.n_joints:
        # the input misses the end effectors, so insert zero vectors at their place
        non_end_effectors = [j for j in range(skeleton.n_joints) if j not in skeleton.end_effectors]
        angles_corr = np.zeros([seq_length, skeleton.n_joints, 3])
        angles_corr[:, non_end_effectors] = angles
        angles = angles_corr
        n_joints = angles.shape[1]

    assert roots.shape[0] == seq_length, 'must have as many root positions as there are frames'
    assert n_joints == skeleton.n_joints, 'unexpected number of joints in skeleton'

    positions = np.zeros_like(angles)  # output positions
    rotations = np.zeros([n_joints, 3, 3])  # must temporarily save rotation matrices for each frame

    # precompute rotation matrices for speedup
    rotmat = exp2rotmat(np.reshape(angles, [-1, 3]))
    rotmat = np.reshape(rotmat, [seq_length, n_joints, 3, 3])

    for f in range(seq_length):
        for j in range(n_joints):

            if skeleton.parents[j] == -1:  # this is the root
                positions[f, j] = skeleton.offsets[j] + roots[f, j]
                rotations[j] = rotmat[f, j]

            else:  # this is a regular joint
                positions[f, j] = np.matmul(skeleton.offsets[j:j+1], rotations[skeleton.parents[j]])
                positions[f, j] += positions[f, skeleton.parents[j]]
                rotations[j] = np.matmul(rotmat[f, j], rotations[skeleton.parents[j]])

    positions = positions[:, :, [0, 2, 1]]  # swap y and z axis
    positions = np.reshape(positions, [seq_length, -1])
    return positions


def padded_array_to_list(data, mask):
    """
    Converts a padded numpy array to a list of un-padded numpy arrays. `data` is expected in shape
    (n, max_seq_length, ...) and `mask` in shape (n, max_seq_length). The returned value is a list of size n, each
    element being an np array of shape (dynamic_seq_length, ...).
    """
    converted = []
    seq_lengths = np.array(np.sum(mask, axis=1), dtype=np.int)
    for i in range(data.shape[0]):
        converted.append(data[i, 0:seq_lengths[i], ...])
    return converted


def export_config(config, output_file):
    """
    Write the configuration parameters into a human readable file.
    :param config: the configuration dictionary
    :param output_file: the output text file
    """
    if not output_file.endswith('.txt'):
        output_file.append('.txt')
    max_key_length = np.amax([len(k) for k in config.keys()])
    with open(output_file, 'w') as f:
        for k in sorted(config.keys()):
            out_string = '{:<{width}}: {}\n'.format(k, config[k], width=max_key_length)
            f.write(out_string)


def export_to_csv(data, ids, output_file):
    """
    Write an array into a csv file.
    :param data: np array of shape (n, seq_length, dof)
    :param ids: array of size n specifying an id for each respective entry in data
    :param output_file: where to store the data
    """
    # treat every frame as a separate sample, otherwise Kaggle goes berserk
    n_samples, seq_length, dof = data.shape
    data_r = np.reshape(data, [-1, dof])

    ids_per_frame = [['{}_{}'.format(id_, i) for i in range(seq_length)] for id_ in ids]
    ids_per_frame = np.reshape(np.array(ids_per_frame), [-1])

    data_frame = pd.DataFrame(data_r,
                              index=ids_per_frame,
                              columns=['dof{}'.format(i) for i in range(dof)])
    data_frame.index.name = 'Id'

    if not output_file.endswith('.csv'):
        output_file.append('.csv')

    data_frame.to_csv(output_file, float_format='%.8f')
