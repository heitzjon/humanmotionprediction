from __future__ import print_function
from __future__ import division

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from utils import forward_kinematics, Skeleton


def visualize_positions(positions, change_color_after_frame=None):
    """
    Visualize motion given 3D positions. Can visualize several motions side by side. If the sequence lengths don't
    match, all animations are displayed until the shortest sequence length.
    :param positions: a list of np arrays in shape (seq_length, n_joints*3) giving the 3D positions per joint and frame.
    :param change_color_after_frame: after this frame id, the color of the plot is changed
    """
    seq_length = np.amin([pos.shape[0] for pos in positions])
    n_joints = positions[0].shape[1]//3
    pos = [np.reshape(p, [-1, n_joints, 3]) for p in positions]
    parents = Skeleton.parents

    # create figure with as many subplots as we have skeletons
    fig = plt.figure()
    axes = [fig.add_subplot(1, len(pos), i + 1, projection='3d') for i in range(len(pos))]
    ax_pred = axes[0]

    # create point object for every bone in every skeleton
    all_lines = []
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']
    for i, joints in enumerate(pos):
        ax = axes[i]
        lines_j = [
            ax.plot(joints[0:1, n,  0], joints[0:1, n, 1], joints[0:1, n, 2], '-o' + colors[i],
                    markersize=3.0)[0] for n in range(1, n_joints)]
        all_lines.append(lines_j)

    # dirty hack to get equal axes behaviour
    min_val = np.amin(pos[0], axis=(0, 1))
    max_val = np.amax(pos[0], axis=(0, 1))
    max_range = (max_val - min_val).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (max_val[0] + min_val[0])
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (max_val[1] + min_val[1])
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (max_val[2] + min_val[2])

    for ax in axes:
        ax.set_aspect('equal')

        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    def on_move(event):
        # find which axis triggered the event
        source_ax = None
        for i in range(len(axes)):
            if event.inaxes == axes[i]:
                source_ax = i
                break

        # transfer rotation and zoom to all other axes
        if source_ax == None:
            return

        for i in range(len(axes)):
            if i != source_ax:
                axes[i].view_init(elev=axes[source_ax].elev, azim=axes[source_ax].azim)
                axes[i].set_xlim3d(axes[source_ax].get_xlim3d())
                axes[i].set_ylim3d(axes[source_ax].get_ylim3d())
                axes[i].set_zlim3d(axes[source_ax].get_zlim3d())
        fig.canvas.draw_idle()

    c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig_text = fig.text(0.05, 0.95, '')

    def update_frame(num, positions, lines, parents, colors):
        for l in range(len(positions)):
            k = 0
            pos = positions[l]
            points_j = lines[l]
            for i in range(1, len(parents)):
                a = pos[num, i]
                b = pos[num, parents[i]]
                p = np.vstack([b, a])
                points_j[k].set_data(p[:, :2].T)
                points_j[k].set_3d_properties(p[:, 2].T)
                if change_color_after_frame and num > change_color_after_frame:
                    points_j[k].set_color(colors[l + 1])
                else:
                    points_j[k].set_color(colors[l])

                k += 1
        time_passed = '{:>.2f} seconds passed'.format(1./25.*num)
        fig_text.set_text(time_passed)

    # create the animation object, for animation to work reference to this object must be kept
    line_ani = animation.FuncAnimation(fig, update_frame, seq_length,
                                       fargs=(pos, all_lines, parents, colors + [colors[0]]),
                                       interval=int(round(1000.0 / 25.0)), blit=False)


    plt.show()


def visualize_joint_angles(joint_angles, change_color_after_frame=None):
    """
    Visualize motion given joint angles in exponential map format.
    :param positions: list of np arrays in shape (seq_length, n_joints*3) giving the 3D positions per joint and frame.
    :param change_color_after_frame: after this frame id, the color of the plot is changed
    """
    positions = [forward_kinematics(ja) for ja in joint_angles]
    visualize_positions(positions, change_color_after_frame)


if __name__ == '__main__':
    # load a random training sequence
    # TODO change data path here
    train_data = np.load('../data/train.npz')['data']
    data = train_data[np.random.randint(len(train_data))]['angles']

    positions = forward_kinematics(data)
    visualize_positions([positions])
