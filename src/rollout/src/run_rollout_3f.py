#!/usr/bin/env python

from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager
import pytransform3d.trajectories as ptr

import rospy
from std_srvs.srv import Empty, EmptyResponse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
import pickle
from rollout.srv import rolloutReq, rolloutReqMod
import time
import glob
from scipy.io import savemat, loadmat
import os
import rospy
from std_msgs.msg import Float64MultiArray, Float32MultiArray, String, Bool, Float32

rollout_srv = rospy.ServiceProxy('/rollout/rollout', rolloutReqMod)

rospy.init_node('run_rollout_set', anonymous=True)

rate = rospy.Rate(30)  # 15hz
state_dim = 16
action_dim = 4

path = 'plans/3f/'

rollout = True
save_graph = True
save_trans = True


def save_display(d, tofile):


    # TODO: state -> obj_pos, obj_rot, act_angles, act_torques
    #                   3       4           3           3

    end = -10
    d = d[:end, :]
    fig = plt.figure(figsize=(10, 10))
    tm = TransformManager()

    for i in range(len(d)):

        object2cam = pt.transform_from_pq(np.hstack((d[i, 0:3],
                                                     pr.quaternion_wxyz_from_xyzw(d[i, 3:7]))))

        tm.add_transform("object"+str(i), "camera", object2cam)


    ax = tm.plot_frames_in("camera", s=0.05, show_name=False)

    ax.set_xlim((-0.1, 0.1))
    ax.set_ylim((-0.1, 0.1))
    ax.set_zlim((0.0, 0.1))

    fig.savefig(tofile + '1')

    fig = plt.figure(figsize=(10, 10))

    fig.add_subplot(3, 1, 1)
    plt.title('Motor angles')
    plt.plot(d[:, 7], 'o', markersize=1)
    plt.plot(d[:, 8], 'o', markersize=1)
    plt.plot(d[:, 9], 'o', markersize=1)

    fig.add_subplot(3, 1, 2)
    plt.title('Motor current')
    plt.plot(d[:, 10], 'o', markersize=1)
    plt.plot(d[:, 11], 'o', markersize=1)
    plt.plot(d[:, 12], 'o', markersize=1)

    fig.add_subplot(3, 1, 3)
    plt.title('Targets')
    plt.plot(d[:, 13], 'o', markersize=1)
    plt.plot(d[:, 14], 'o', markersize=1)
    plt.plot(d[:, 15], 'o', markersize=1)

    fig.savefig(tofile)


if rollout:

    # TODO: state -> obj_pos, obj_quat, act_angles, act_torques, targets
    #                   3       4           3           3,       3

    files = glob.glob(path + "*.txt")
    logdir_prefix = 'rollout'
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    for i in range(len(files)):

        action_file = files[i]
        npfile = action_file[:-4]
        logdir = logdir_prefix + '_' + npfile
        index = time.strftime("%d-%m-%Y_%H-%M-%S")
        logdir = os.path.join(data_path, logdir)

        if not (os.path.exists(logdir)):
            os.makedirs(logdir)

        print("\nLOGGING TO: " + logdir + "\n")

        print('Rolling-out file number ' + str(i + 1) + ': ' + action_file + '.\n')

        A = np.loadtxt(action_file, delimiter=',', dtype=float)[:, :]

        Af = A.reshape((-1,))
        Af /= 5

        for j in range(1):
            # conduct j numbers of each rollout
            rospy.sleep(2)

            resp = rollout_srv(Af)

            S = np.array(resp.states).reshape(-1, state_dim)
            NS = np.array(resp.next_states).reshape(-1, state_dim)
            A = np.array(resp.actions).reshape(-1, action_dim)
            T = np.array(resp.time).reshape(-1, 1)
            TP = np.array(resp.target_pos).reshape(-1, action_dim)

            rospy.sleep(2)

            transition = {"observation": S[:, :],
                          "action": A[:, :],
                          "target_pos": TP[:, :],
                          "next_observation": NS[:, :],
                          "time": T[:],
                          "success": resp.success,
                          "reason": resp.reason}

            print('suc: ' + resp.reason + '\tnum_actions: ' + str(len(A)))

            if len(S) <= 10:
                continue

            file_name = 'states_' + index + '_itr_' + str(j)
            np.save(logdir + '/' + file_name + '.npy', S)

            if save_graph:
                graphs_dir = logdir + '/graphs'
                if not (os.path.exists(graphs_dir)):
                    os.makedirs(graphs_dir)
                save_display(S, graphs_dir + '/' + file_name)

            if save_trans:
                trans = logdir + '/transition'
                if not (os.path.exists(trans)):
                    os.makedirs(trans)
                with open(trans + '/' + file_name + '.pickle', 'wb') as handle:
                    pickle.dump(transition, handle, protocol=pickle.HIGHEST_PROTOCOL)