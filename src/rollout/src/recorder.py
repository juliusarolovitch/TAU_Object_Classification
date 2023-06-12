#!/usr/bin/env python

'''
Author:  Osher Azulay
'''

import rospy
from std_msgs.msg import String, Float32MultiArray, Bool, Float32
from std_srvs.srv import Empty, EmptyResponse, SetBool
from rollout.srv import rolloutReq, rolloutReqFile, plotReq, observation, IsDropped, TargetAngles, gets
from hand_control.srv import RegraspObject, close
import numpy as np
from pose_node.srv import one_predict
import matplotlib.pyplot as plt
import pickle
import geometry_msgs.msg
from sensor_msgs.msg import CompressedImage, Image
import os
from cv_bridge import CvBridge, CvBridgeError
import cv2
import time
from cl_control.srv import pathTrackReq, PlanName, PlanNameResponse
import matplotlib.font_manager as fm
import matplotlib as mpl
font_names = [f.name for f in fm.fontManager.ttflist]
import matplotlib.font_manager as fm
import subprocess
from matplotlib.patches import Rectangle

from matplotlib.patches import Circle
# Rebuild the matplotlib font cache
fm._rebuild()
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy.random as rnd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.collections import PatchCollection
# Edit the font, font size, and axes width
mpl.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 1
pred_color = 'black'
gt_color = 'dodgerblue'
gt_color_px = 'fuchsia'
np.set_printoptions(2)
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy.random as rnd
import numpy as np
import matplotlib as mpl

display_images = True
display_prediction = True
display_obs = False
record = False
display_gt = True

class RotatingRectangle():
    def __init__(self, xy, width, height, rel_point_of_rot,angle, color, alpha):
        self.rect = mpl.patches.Rectangle(xy=(xy), width= width, height=height, color=color, alpha=alpha, angle = angle,fill=False)
        # Rectangle.__init__(xy, width, height, **kwargs)
        self.rel_point_of_rot = rel_point_of_rot
        self.xy_center = self.rect.get_xy()
        self.angle = angle
        self.set_angle(self.angle)

    def _apply_rotation(self):
        angle_rad = self.angle * np.pi / 180
        m_trans = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                            [np.sin(angle_rad), np.cos(angle_rad)]])

        shift = np.matmul(-m_trans, self.rel_point_of_rot)
        self.rect.set_xy(self.xy_center + shift)

    def set_angle(self, angle):
        self.angle = angle
        self._apply_rotation()

    def set_rel_point_of_rot(self, rel_point_of_rot):
        self.rel_point_of_rot = rel_point_of_rot
        self._apply_rotation()

    def set_xy_center(self, xy):
        self.xy_center = xy
        self._apply_rotation()

def arc_patch(xy, width, height, theta1=0., theta2=180., resolution=50, **kwargs):

    # generate the points
    theta = np.linspace(np.radians(theta1), np.radians(theta2), resolution)
    points = np.vstack((width*np.cos(theta) + xy[0],
                        height*np.sin(theta) + xy[1]))
    # build the polygon and add it to the axes
    poly = mpatches.Polygon(points.T, closed=False, **kwargs)

    return poly

def crop(img):
    return img#img[200:-5,50:710, :]

def setgrid(ax):
    ax.tick_params(axis='y', which='left', labelleft=False, labelright=True)
    ax.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)
    ax.minorticks_on()
    ax.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)



class rolloutRecorder():

    # simple record node
    # require trigger - /rollout/record_trigger
    # launch a service to get state

    running = False
    suc = True
    fail = False
    display = True
    ref_flag = False
    clear = False
    drop = True
    grasped = False

    action = np.array([0., 0.])  # left, right
    obj_pos = np.array([0., 0., 0.])  # x, y, theta
    obj_pos_pixels = np.array([0,0])
    obj_pos_nn = np.array([0., 0., 0.])  # x, y, theta
    start_pos = np.array([0., 0., 0.])
    goal = np.array([0., 0., 0.])
    object_true_traj = np.array([0., 0., 0.])
    object_pred_traj = np.array([0., 0., 0.])
    object_true_traj_px = np.array([0., 0.])
    obs = np.array([0., 0., 0., 0., 0., 0.])
    S, A, T = [], [], []
    images, Timages = [], []
    I = np.zeros((480, 640, 3), dtype=np.uint8)

    xy_gt = np.array([0., 0.])
    xy_pred = np.array([0., 0.])

    T0 = 0.
    counter = 0
    drop_counter = 0
    start_rot = 0

    def __init__(self):
        rospy.init_node('rollout_recorder', anonymous=True)

        rospy.Subscriber('/hand_control/drop', Bool, self.callbackObjectDrop)
        rospy.Subscriber('/rollout/action', Float32MultiArray, self.callbackAction)
        rospy.Subscriber('/rollout/fail', Bool, self.callbacFail)

        rospy.Service('/recorder/record_trigger', SetBool, self.callbackTrigger)
        rospy.Service('/recorder/save_rollout', SetBool, self.callbackSave)
        rospy.Service('/recorder/get_states', gets, self.callbackStates)

        obs_srv = rospy.ServiceProxy('/observation', observation)
        if display_prediction:
            self.pose_predict = rospy.ServiceProxy('/pose_nn/predict', one_predict)
            rospy.Subscriber('/nn/obj_pos', Float32MultiArray, self.callbackPredictedPos)

        self.params = np.array([rospy.get_param('/openhandNode/motor_offset')[0],
                                rospy.get_param('/openhandNode/motor_offset')[1],
                                rospy.get_param('/hand_control/finger_close_load')])

        rospy.Subscriber('/control/goal', Float32MultiArray, self.callbackGoal)
        rospy.Subscriber('/hand_control/obj_pos_pixels', Float32MultiArray, self.callbackPixels)
        rospy.Subscriber('/rollout/ObjectIsReset', String, self.graspStatusCallback)

        rospy.Service('/plot/clear', Empty, self.callbackClear)
        rospy.Service('/plot/ref', pathTrackReq, self.callbackPlot)

        if display_images:
            self.bridge = CvBridge()
            rospy.Subscriber('/camera/image_raw', Image, self.callbackImage, queue_size=1)

        self.config_display(False)

        print('[recorder] Ready ...')

        if not display_images:
            self.rate = rospy.Rate(10)  # 10hz
        else:
            self.rate = rospy.Rate(10)  # 2hz

        while not rospy.is_shutdown():

            if True: #if self.running and not self.drop and self.grasped:

                self.counter += 1
                self.state = np.array(obs_srv().state)

                if not self.start_rot:
                    self.start_rot = 1 * self.state[2]
                    self.start_rot_obj = 0 * self.state[2] % (np.pi / 4.)  # 4 for sqr, without for cyl

                self.state[2] = self.state[2] - self.start_rot + self.start_rot_obj

                if self.state[2] > np.pi: self.state[2] -= 2. * np.pi
                if self.state[2] < -np.pi: self.state[2] += 2. * np.pi

                self.state[:2] *= 1000  # convert to mm
                self.state[2] *= 180.0 / np.pi  # convert to degrees

                self.obj_pos = self.state[:3]
                if display_prediction:
                    self.object_pred_traj = np.vstack((self.object_pred_traj, self.obj_pos_nn))
                    # self.object_pred_traj[-1, 2] = -1*self.object_pred_traj[-1, 2] # ????????

                if display_gt:
                    self.object_true_traj = np.vstack((self.object_true_traj, self.obj_pos))
                    # self.object_true_traj[-1, 2] = -1*self.object_true_traj[-1, 2]
                    self.object_true_traj_px = np.vstack((self.object_true_traj_px, self.obj_pos_pixels))

                if self.counter <= 5: self.start_pos = self.obj_pos

                self.S.append(self.state)
                self.A.append(self.action)
                self.T.append(rospy.get_time() - self.T0)

                if display_images and self.counter % 2:
                    self.images.append(self.I)
                    self.Timages.append(rospy.get_time() - self.T0)

                if self.fail:
                    print('[recorder] Episode ended with %d points.' % len(self.S))
                    if record: self.p.communicate()
                    self.running = False

                # if self.counter % 2 == 0:
                self.update_plot(False)

            self.rate.sleep()

    def config_display(self, blit):
        plt.close('all')

        mpl.rcParams['font.family'] = 'DejaVu Serif'
        plt.rcParams['font.size'] = 8
        plt.rcParams['axes.linewidth'] = 1

        self.fig = plt.figure(figsize=(16, 8))
        canvas_width, canvas_height = self.fig.canvas.get_width_height()
        self.ax1 = self.fig.add_subplot(1, 2, 1)
        self.ax2 = self.fig.add_subplot(1, 2, 2)
        self.ax1.set_yticklabels([])
        self.ax1.set_xticklabels([])
        for axis in ['top', 'bottom', 'left', 'right']:
            self.ax1.spines[axis].set_linewidth(10)
        # obs
        if display_obs:
            self.ax3 = self.fig.add_subplot(5, 3, 10)
            self.ax4 = self.fig.add_subplot(5, 3, 11)
            self.ax5 = self.fig.add_subplot(5, 3, 12)
            self.ax6 = self.fig.add_subplot(5, 3, 13)
            self.ax7 = self.fig.add_subplot(5, 3, 14)
            self.ax8 = self.fig.add_subplot(5, 3, 15)

            self.ax3.set_xticklabels([]), self.ax4.set_xticklabels([]), self.ax5.set_xticklabels([])
            self.ax3.set_yticklabels([]), self.ax4.set_yticklabels([]), self.ax5.set_yticklabels([])
            self.ax6.set_xticklabels([]), self.ax7.set_xticklabels([]), self.ax8.set_xticklabels([])
            self.ax6.set_yticklabels([]), self.ax7.set_yticklabels([]), self.ax8.set_yticklabels([])

        if display_images:
            self.img = self.ax1.imshow(self.I)

        if display_prediction:
            self.line_pred, = self.ax2.plot([], linewidth=2, alpha=0.2, color=pred_color)
            self.ori_pred, = self.ax2.plot([], linestyle='--', linewidth=1, alpha=0.5, color=pred_color,)
            self.pred_circle, = self.ax2.plot([], color=pred_color, marker='o', alpha=0.5, markersize=60,fillstyle='none')
            self.marker_pred, = self.ax2.plot([], color=pred_color, alpha=1,marker='o', markersize=2, label='Predicted')

        if display_gt:
            self.line_gt, = self.ax2.plot([], color=gt_color, linewidth=2, alpha=0.3)
            self.marker_gt, = self.ax2.plot([], color=gt_color, alpha=1, marker='o', markersize=2)
            self.ori_gt, = self.ax2.plot([], color=gt_color, alpha=1, linewidth=2, linestyle='--',)
            self.gt_circle, = self.ax2.plot([], color=gt_color,marker='o', alpha=0.7, markersize=60,fillstyle='none')

            self.line_gt_px, = self.ax1.plot([], color=gt_color_px, linewidth=2, alpha=0.4)
            self.marker_gt_px, = self.ax1.plot([], color=gt_color_px, alpha=1, marker='o', markersize=2)


        self.goal_plot, = self.ax2.plot([], color='green',marker='x', alpha=0.8, markersize=6 ,)
        self.goal_plot_px,= self.ax1.plot([], color='green',marker='x', alpha=0.8, markersize=6 ,)

        if self.ref_flag:
            self.ref_path, = self.ax2.plot([], '--g',linewidth=1, alpha=0.4, label='Reference Trajectory')
            self.ref_path_px, = self.ax1.plot([], '--g',linewidth=1, alpha=0.4, label='Reference Trajectory')

        self.ax2.set_xlim(-85, 85)
        self.ax2.set_ylim(65, 150)
        self.ax2.set_xlabel('X (mm)')
        self.ax2.set_ylabel('Y (mm)')
        self.ax2.set_aspect('equal', 'box')
        setgrid(self.ax2)

        if display_obs:
            one_c = 'navy'
            two_c = 'lightsteelblue'

            self.angle_plot1, = self.ax3.plot([], color=one_c, alpha=0.9, linewidth=2)
            self.angle_plot2, = self.ax6.plot([], color=two_c, alpha=0.9, linewidth=2)
            self.load_plt1, = self.ax4.plot([], color=one_c, alpha=0.9, linewidth=2)
            self.load_plt2, = self.ax7.plot([], color=two_c, alpha=0.9, linewidth=2)
            self.force_plt1, = self.ax5.plot([], color=one_c, alpha=0.9, linewidth=2, label='Finger 1')
            self.force_plt2, = self.ax8.plot([], color=two_c, alpha=0.9, linewidth=2,label='Finger 2')

            setgrid(self.ax3), setgrid(self.ax4), setgrid(self.ax5), setgrid(self.ax6),setgrid(self.ax7), setgrid(self.ax8)

            self.ax3.set_title('Motor angle')
            self.ax4.set_title('Motor load')
            self.ax5.set_title('Finger force')
            # self.ax3.set_ylim(-0.01, 1)
            # self.ax4.set_ylim(-150, 150)
            # self.ax5.set_ylim(-100, 400)
            lines_labels = [ax.get_legend_handles_labels() for ax in self.fig.axes[2:]]
            lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
            self.fig.legend(lines, labels, loc='lower center', ncol=2)

        self.ax2.legend()
        self.fig.canvas.draw()

        if blit:
            # cache the background
            self.axbackground = self.fig.canvas.copy_from_bbox(self.ax1.bbox)
            self.ax2background = self.fig.canvas.copy_from_bbox(self.ax2.bbox)
            if display_obs:
                self.ax3background = self.fig.canvas.copy_from_bbox(self.ax3.bbox)
                self.ax4background = self.fig.canvas.copy_from_bbox(self.ax4.bbox)
                self.ax5background = self.fig.canvas.copy_from_bbox(self.ax5.bbox)
                self.ax6background = self.fig.canvas.copy_from_bbox(self.ax6.bbox)
                self.ax7background = self.fig.canvas.copy_from_bbox(self.ax7.bbox)
                self.ax8background = self.fig.canvas.copy_from_bbox(self.ax8.bbox)

        plt.show(block=False)

        if record:
            # Open an ffmpeg process
            outf = '/home/osher/catkin_ws/src/manipulations/cl_control/src/ffmpeg.mp4'
            cmdstring = ('ffmpeg',
                         '-y', '-r', '10',  # overwrite, 30fps
                         '-s', '%dx%d' % (canvas_width, canvas_height),  # size of image string
                         '-pix_fmt', 'argb',  # format
                         '-f', 'rawvideo', '-i', '-',  # tell ffmpeg to expect raw video from the pipe
                         '-vcodec', 'png', outf)  # output encoding
            self.p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE)

    def update_plot(self, blit):
        r = 1000 * 0.02 / 2

        if display_images:
            self.img.set_data(self.I)

        if display_gt:

            # obj3 = arc_patch(xy=(-self.object_true_traj[-1, 0],self.object_true_traj[-1, 1] ), width=1.5*r,
            #                 height=1.5*r, theta1=self.object_true_traj[-1, 2], theta2=self.object_true_traj[-1, 2] + 180., fill=False)
            # xy = obj3.xy[obj3.xy.shape[0] / 2]
            # self.xy_gt = np.vstack((self.xy_gt, xy))
            # self.line_gt.set_data(self.xy_gt[1:-1,0],
            #                        self.xy_gt[1:-1,1]) # plot obj gt pos
            # self.marker_gt.set_data(self.xy_gt[-1,0],
            #                         self.xy_gt[-1,1])
            self.line_gt.set_data(self.object_true_traj[1:-1, 0], self.object_true_traj[1:-1, 1])
            self.marker_gt.set_data(self.object_true_traj[-1, 0], self.object_true_traj[-1, 1])

            self.line_gt_px.set_data(self.object_true_traj_px[1:-1, 0], self.object_true_traj_px[1:-1, 1])
            self.marker_gt_px.set_data(self.object_true_traj_px[-1, 0], self.object_true_traj_px[-1, 1])
            ####################
            ### Objects
            ###################
            # self.gt_circle.set_data(self.object_true_traj[-1, 0], self.object_true_traj[-1, 1])
            # obj = mpl.patches.Ellipse(xy=(-self.object_true_traj[-1, 0],self.object_true_traj[-1, 1]), width=20, height=40, angle= 180 + self.object_true_traj[-1, 2],  fill=False, color=gt_color, alpha = 0.5)
            # obj = RotatingRectangle((-self.object_true_traj[-1, 0], self.object_true_traj[-1, 1]), width=20, height=20,
            #                         rel_point_of_rot=([10,10]),
            #                         angle=self.object_true_traj[-1, 2], color=gt_color, alpha=0.5)

            # obj = mpl.patches.Ellipse(xy=(-self.object_true_traj[-1, 0],self.object_true_traj[-1, 1]),
            #                           width=2*r, height=2*r, angle= 180 + self.object_true_traj[-1, 2],  fill=True, color=gt_color, alpha = 0.5)
            # obj2 = mpl.patches.Ellipse(xy=(-self.object_true_traj[-1, 0] - r*np.cos(self.object_true_traj[-1, 2] * np.pi / 180), self.object_true_traj[-1, 1] - r*np.sin(self.object_true_traj[-1, 2] * np.pi / 180)),
            #                            width=2*r, height=2*r, angle= 180 + self.object_true_traj[-1, 2],  fill=True, color='white', alpha = 1.0)

            # obj = Arc((-self.object_true_traj[-1, 0] , self.object_true_traj[-1, 1]),
            #           2*r, 2*r, self.object_true_traj[-1, 2], 0.0, 180,color=gt_color, alpha=0.5)
            # obj2 = Arc((-self.object_true_traj[-1, 0],self.object_true_traj[-1, 1] ),2*r, 4*r, self.object_true_traj[-1, 2], 0.0, 180,color=gt_color, alpha=0.5)
            #
            # self.ax2.add_patch(obj) # obj.rect
            # self.ax2.add_patch(obj2) # obj.rect
            arrow1 = mpatches.Arrow(self.object_true_traj[-1, 0], self.object_true_traj[-1, 1], r*np.cos(self.object_true_traj[-1, 2] * np.pi / 180), r*np.sin(self.object_true_traj[-1, 2] * np.pi / 180),width=r/2, color=gt_color)
            arrow2 = mpatches.Arrow(self.object_true_traj[-1, 0], self.object_true_traj[-1, 1], r*np.cos((self.object_true_traj[-1, 2]- 270) * np.pi / 180), r*np.sin((self.object_true_traj[-1, 2]-270) * np.pi / 180),width=r/2, color=gt_color)

            # x_values = [self.object_true_traj[-1, 0], self.object_true_traj[-1, 0] + r*np.cos(self.object_true_traj[-1, 2] * np.pi / 180)]
            # y_values = [self.object_true_traj[-1, 1], self.object_true_traj[-1, 1] + r*np.sin(self.object_true_traj[-1, 2] * np.pi / 180)]
            # self.ori_gt.set_data(x_values, y_values)
            self.ax2.add_patch(arrow1) # obj.rect
            self.ax2.add_patch(arrow2) # obj.rect

        if self.goal.any():
          self.goal_plot.set_data(self.goal[0] * 1000, self.goal[1] * 1000)
          if self.ref_flag:
             self.ref_path.set_data(self.Sref[:, 0], self.Sref[:, 1])

        if display_prediction:
            # obj3 = arc_patch(xy=(-self.object_pred_traj[-1, 0],self.object_pred_traj[-1, 1] ), width=1.5*r,
            #                 height=1.5*r, theta1=self.object_true_traj[-1, 2], theta2=self.object_true_traj[-1, 2] + 180., fill=False)
            # xy = obj3.xy[obj3.xy.shape[0] / 2]
            # self.xy_pred = np.vstack((self.xy_pred, xy))
            # self.line_pred.set_data(self.xy_pred[1:-1,0],
            #                        self.xy_pred[1:-1,1]) # plot obj gt pos
            # self.marker_pred.set_data(self.xy_pred[-1,0],
            #                         self.xy_pred[-1,1])

            self.line_pred.set_data(self.object_pred_traj[1:-1, 0], self.object_pred_traj[1:-1, 1])
            self.marker_pred.set_data(self.object_pred_traj[-1, 0], self.object_pred_traj[-1, 1])

            # obj = plt.Circle((-self.object_pred_traj[-1, 0], self.object_pred_traj[-1, 1]), r, fill=False, color=pred_color, alpha = 0.5)
            # obj = mpl.patches.Ellipse(xy=(-self.object_pred_traj[-1, 0], self.object_pred_traj[-1, 1]), width=20,
            #                           height=40, angle=180 + self.object_pred_traj[-1, 2], fill=False, color=pred_color,
            #                           alpha=0.5)
            # obj = mpl.patches.Ellipse(xy=(-self.object_pred_traj[-1, 0],self.object_pred_traj[-1, 1]),
            #                           width=2*r, height=2*r, angle= 180 + self.object_pred_traj[-1, 2],  fill=True, color=pred_color, alpha = 0.5)
            # obj2 = mpl.patches.Ellipse(xy=(-self.object_pred_traj[-1, 0] - r*np.cos(self.object_pred_traj[-1, 2] * np.pi / 180), self.object_pred_traj[-1, 1] - r*np.sin(self.object_pred_traj[-1, 2] * np.pi / 180)),
            #                            width=2*r, height=2*r, angle= 180 + self.object_pred_traj[-1, 2],  fill=True, color='white', alpha = 1.0)

            # obj = Arc((-self.object_pred_traj[-1, 0] ,self.object_pred_traj[-1, 1] ),2*r, 2*r,self.object_pred_traj[-1, 2], 0.0, 180,color=pred_color, alpha=0.5)
            # obj2 = Arc((-self.object_pred_traj[-1, 0],self.object_pred_traj[-1, 1] ),2*r, 4*r,self.object_pred_traj[-1, 2], 0.0, 180,color=pred_color, alpha=0.5)
            #
            # self.ax2.add_patch(obj) # obj.rect
            # self.ax2.add_patch(obj2) # obj.rect

            # obj = RotatingRectangle((-self.object_pred_traj[-1, 0], self.object_pred_traj[-1, 1]), width=20, height=20,
            #                         rel_point_of_rot=([10, 10]),
            #                         angle=self.object_pred_traj[-1, 2], color=pred_color, alpha=0.5)


            # x_values = [-self.object_pred_traj[-1, 0],
            #             self.object_pred_traj[-1, 0] + r * np.cos(self.object_pred_traj[-1, 2] * np.pi / 180)]
            # y_values = [-self.object_pred_traj[-1, 1],
            #             self.object_pred_traj[-1, 1] + r * np.sin(self.object_pred_traj[-1, 2] * np.pi / 180)]
            #
            # self.ori_pred.set_data(x_values, y_values)
            # self.pred_circle.set_data(self.object_pred_traj[-1, 0], self.object_pred_traj[-1, 1])
            arrow1 = mpatches.Arrow(self.object_pred_traj[-1, 0], self.object_pred_traj[-1, 1], r*np.cos(self.object_pred_traj[-1, 2] * np.pi / 180), r*np.sin(self.object_pred_traj[-1, 2] * np.pi / 180),width=r/2, color=pred_color, alpha=0.7)
            arrow2 = mpatches.Arrow(self.object_pred_traj[-1, 0], self.object_pred_traj[-1, 1], r*np.cos((self.object_pred_traj[-1, 2]- 270) * np.pi / 180), r*np.sin((self.object_pred_traj[-1, 2]-270) * np.pi / 180),width=r/2, color=pred_color, alpha=0.7)
            self.ax2.add_patch(arrow1) # obj.rect
            self.ax2.add_patch(arrow2) # obj.rect

        # obs
        if display_obs:
            self.force_plt1.set_data(range(len(self.obs[1:, 4])), self.obs[1:, 4])
            self.force_plt2.set_data(range(len(self.obs[1:, 5])),  self.obs[1:, 5])

            self.load_plt1.set_data(range(len(self.obs[1:, 2])), self.obs[1:, 2])
            self.load_plt2.set_data(range(len(self.obs[1:, 3])), self.obs[1:, 3])

            self.angle_plot1.set_data(range(len(self.obs[1:, 0])), self.obs[1:, 0])
            self.angle_plot2.set_data(range(len(self.obs[1:, 1])), self.obs[1:, 1])


        if blit:
            self.fig.canvas.restore_region(self.axbackground)
            self.fig.canvas.restore_region(self.ax2background)

            if display_obs:
                self.fig.canvas.restore_region(self.ax3background)
                self.fig.canvas.restore_region(self.ax4background)
                self.fig.canvas.restore_region(self.ax5background)
                self.fig.canvas.restore_region(self.ax6background)
                self.fig.canvas.restore_region(self.ax7background)
                self.fig.canvas.restore_region(self.ax8background)

            # redraw just the points
            if display_images:
                self.ax1.draw_artist(self.img)

            self.ax2.draw_artist(self.line_gt)
            self.ax2.draw_artist(self.ori_gt)
            self.ax2.draw_artist(self.gt_circle)
            self.ax2.draw_artist(self.marker_gt)

            if self.goal.any():
                self.ax2.draw_artist(self.goal_plot)
                if self.ref_flag:
                    self.ax2.draw_artist(self.ref_path)

            if display_prediction:
                self.ax2.draw_artist(self.line_pred)
                self.ax2.draw_artist(self.ori_pred)
                self.ax2.draw_artist(self.pred_circle)
                self.ax2.draw_artist(self.marker_pred)

            if display_obs:
                self.ax3.draw_artist(self.angle_plot1)
                self.ax6.draw_artist(self.angle_plot2)
                self.ax4.draw_artist(self.load_plt1)
                self.ax7.draw_artist(self.load_plt2)
                self.ax5.draw_artist(self.force_plt1)
                self.ax8.draw_artist(self.force_plt2)

            # fill in the axes rectangle
            self.fig.canvas.blit(self.ax1.bbox)
            self.fig.canvas.blit(self.ax2.bbox)

            if display_obs:
                self.fig.canvas.blit(self.ax3.bbox)
                self.fig.canvas.blit(self.ax4.bbox)
                self.fig.canvas.blit(self.ax5.bbox)
                self.fig.canvas.blit(self.ax6.bbox)
                self.fig.canvas.blit(self.ax7.bbox)
                self.fig.canvas.blit(self.ax8.bbox)

        else:
            self.fig.canvas.draw()

        if display_obs:
            self.ax3.relim(), self.ax4.relim(), self.ax5.relim()
            self.ax6.relim(), self.ax7.relim(), self.ax8.relim()
            self.ax3.autoscale(True, 'both', False), self.ax4.autoscale(True, 'both', False), self.ax5.autoscale(True, 'both', False)
            self.ax6.autoscale(True, 'both', False), self.ax7.autoscale(True, 'both', False), self.ax8.autoscale(True, 'both', False)

        self.fig.canvas.flush_events()
        # plt.pause(0.000000000001)
        self.ax2.patches = []

        if record:
            string = self.fig.canvas.tostring_argb()
            self.p.stdin.write(string)

    def callbackObjectDrop(self, msg):

        if msg.data:
            self.drop_counter += 1
        else:
            self.drop_counter = 0

        if msg.data and self.drop_counter >= 3:
            self.drop = True
        else:
            self.drop = False

    def callbackClear(self, msg):
        self.clear = True
        return EmptyResponse()

    def callbackGoal(self, msg):
        self.goal = np.array(msg.data)

    def callbackPixels(self, msg):
        self.obj_pos_pixels = np.array(msg.data)

    def graspStatusCallback(self,msg):
        if msg.data == "finished":
            self.grasped = True
        else:
            self.grasped = False

    def callbackPredictedPos(self, msg):
        if self.counter < 5:
            self.obj_pos_nn = np.array(msg.data)
        else:
            e = 0.9
            self.obj_pos_nn = np.array(msg.data) * (1 - e) + e * self.obj_pos_nn


    def callbackPlot(self, req):
        self.Sref = np.array(req.desired_path).reshape(-1,2)
        self.Sref *= 1000
        self.ref_flag = True if self.Sref.shape[0] > 5 else False

        return {'real_path': [], 'success' : True}

    def callbackAction(self, msg):
        self.action = np.array(msg.data)

    def callbacFail(self, msg):
        self.fail = msg.data

    def callbackImage(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
        frame = np.array(frame, dtype=np.uint8)
        self.I = crop(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def callbackTrigger(self, msg):
        self.running = msg.data
        if self.running:
            self.counter = 0
            self.S = []
            self.A = []
            self.T = []
            self.T0 = rospy.get_time()
            self.images = []
            self.Timages = []
            self.object_true_traj = np.array([[0., 0., 0.]])
            self.object_true_traj_px = np.array([[0., 0.]])
            self.object_pred_traj = np.array([0., 0., 0.])
            self.obs = np.array([0., 0., 0., 0., 0., 0.])
            self.start_rot = 0.0
            print('[recorder] Recording started.')
            self.fail = False
        else:
            if False:
                self.callbackSave(True)
            print('[recorder] Recording stopped with %d points.' % len(self.S))

        return {'success': True, 'message': ''}

    def callbackSave(self, msg):
        name = str(self.goal)
        print('SAVE REQUEST')
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../sample_trajectories')
        if not (os.path.exists(path)):
            os.makedirs(path)
        file_name = name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
        print('data length\t' + str(len(self.S)) + '\n num images:\t' + str(len(self.images)))

        trajectory = {"observation": self.S,
                      "time": self.T,
                      "action": self.A,
                      "true_traj": self.object_true_traj,
                      "pred_traj": self.object_pred_traj,
                      "true_traj_px": self.object_true_traj_px,
                      # "image": self.images,
                      # "Timage": self.Timages,
                      }

        # trajectory = {"observation": self.S,
        #                       "time": self.T,
        #                       "action": self.A,
        #                       "true_traj": self.object_true_traj,
        #                       "pred_traj": self.object_pred_traj,
        #                       "true_traj_px": self.object_true_traj_px,
        #                       "pred_traj_px": self.object_pred_traj_px,
        #                       "image": self.images,
        #                       "Timage": self.Timages,
        #                       "Sref": self.Sref
        #                       }

        with open(path + '/' + file_name + '.pkl', 'wb') as f:
            pickle.dump(trajectory, f)

        return {'success': True, 'message': ''}

    def callbackStates(self, msg):
        return {'states': np.array(self.S).reshape((-1,)),
                'actions': np.array(self.A).reshape((-1,)),
                'time': np.array(self.T).reshape((-1,))}



if __name__ == '__main__':
    try:
        import matplotlib.pyplot as plt
        rolloutRecorder()
    except rospy.ROSInterruptException:
        pass


