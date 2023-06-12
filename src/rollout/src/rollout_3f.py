#!/usr/bin/env python

'''
Updated by Osher Azulay
'''

import rospy
from past.builtins import raw_input
from std_msgs.msg import String, Float32MultiArray, Bool
from std_srvs.srv import Empty, EmptyResponse, SetBool
from rollout.srv import rolloutReqMod, rolloutReqFile, observation, IsDropped, TargetAngles, gets
from hand_control.srv import RegraspObject, close, TargetPos
import numpy as np
import matplotlib.pyplot as plt
import pickle


class rolloutPublisher():


    obj_pos = np.array([0., 0., 0.])
    obj_rot = np.array([0., 0., 0., 0. ])
    reason = ''
    reset_status = 'waiting'
    drop, suc, fail, trigger, running = True, True, True, True, False
    S, A, T, NS, TP = [], [], [], [], []
    start_rot, start_rot_obj, T0 = 0., 0., 0.
    drop_counter = 0

    object_map = {'circle': 0,
                   'hexagon': 3,
                   'ellipse': 4,
                   'rectangle': 5,
                   'square': 6,
                   'box': 9,
                   'wire': 11,
                   'gum': 10,
                   'star': 7}

    def __init__(self):

        rospy.init_node('rollout', anonymous=True)

        rospy.Service('/rollout/rollout', rolloutReqMod, self.CallbackRollout)
        rospy.Service('/rollout/rollout_from_file', rolloutReqFile, self.CallbackRolloutFile)
        rospy.Service('/rollout/run_trigger', SetBool, self.callbackStop)  # stop trigger

        self.action_pub = rospy.Publisher('/rollout/action', Float32MultiArray, queue_size=10)  # publish the action
        self.fail_reason = rospy.Publisher('/rollout/fail_reason', String, queue_size=10)

        rospy.Subscriber('/rollout/ObjectIsReset', String, self.callbackTrigger)
        rospy.Subscriber('/hand_control/obj_relative_pos', Float32MultiArray, self.callbackObj)
        rospy.Subscriber('/hand_control/obj_relative_quat', Float32MultiArray, self.callbackObjQ)

        rospy.Subscriber('/hand_control/drop', Bool, self.callbackObjectDrop)

        # gripper services
        self.regrasp_srv = rospy.ServiceProxy('/RegraspObject', RegraspObject)
        self.drop_srv = rospy.ServiceProxy('/IsObjDropped', IsDropped)
        self.move_srv = rospy.ServiceProxy('/MoveGripper', TargetAngles)
        self.obs_srv = rospy.ServiceProxy('/observation', observation)
        self.gets_srv = rospy.ServiceProxy('/rollout/get_states', gets)
        self.open_srv = rospy.ServiceProxy('/OpenGripper', Empty)
        self.close_srv = rospy.ServiceProxy('/CloseGripper', close)
        self.target_pos_srv = rospy.ServiceProxy('/TargetPos', TargetPos)

        self.state_dim = 16
        self.action_dim = 4
        self.stepSize = 1

        print('[rollout] Ready to rollout...')

        self.rate = rospy.Rate(30)  # 15hz
        rospy.spin()

    def run_rollout(self, A):

        self.trigger = False
        # self.reset()
        self.close_srv()
        self.fail = False
        self.running = True
        success = True

        print("[rollout] Rolling-out actions...")

        step_counter = 0
        i = 0
        action = [0, 0, 0, 0]

        # TODO: state -> obj_pos, obj_quat, act_angles, act_torques
        #                   3       4           3           3

        state = np.array(self.obs_srv().state)

        st = rospy.get_time()

        while self.running:

            self.S.append(np.copy(state))

            if step_counter <= 0:

                if i > 2:
                    # action = A[i, :] * np.random.choice(range(0, 2), 2) * 0.1
                    # step_counter = self.stepSize
                    action = np.random.uniform(low=-0.005, high=+0.03, size=4)
                    action[0] = 0
                    step_counter = np.random.randint(5)
                else:
                    action = A[i, :] * 0.1  # * 0.1
                    step_counter = self.stepSize
                    # if not i % 10:
                    #     action = np.random.uniform(low=-0.005, high=+0.01, size=2)
                    #

                i += 1

            print('[rollout] Applying action:  {} \ttime: {}'.format(action, rospy.get_time() - st))

            self.A.append(np.copy(action))
            self.TP.append(np.array(self.target_pos_srv().angle))  # TODO might be crucial
            self.T.append(rospy.get_time() - self.T0)

            self.suc = self.move_srv(action).success

            next_state = np.array(self.obs_srv().state)
            self.NS.append(np.copy(next_state))

            state = np.copy(next_state)

            step_counter -= 1

            if not self.suc:
                print("[rollout] Load Fail")
                self.running = False
                success = False
                self.fail_reason.publish('load')
                self.reason = 'load'
                break

            # elif self.drop:
            #     print("[rollout] Drop Fail")
            #     self.running = False
            #     success = False
            #     self.fail_reason.publish('drop')
            #     self.reason = 'drop'
            #     break

            if i >= A.shape[0] and step_counter == 0:
                print("[rollout] Complete.")
                success = True
                self.reason = 'ok'
                break

            self.rate.sleep()


        rospy.sleep(1.)
        # self.open_srv()

        return success

    def reset(self):
        rospy.logerr("[rollout] Regrasping the object")
        while 1:
            if not self.trigger:
                rospy.logwarn('[rollout] Waiting for arm to grasp object...')
                self.regrasp_srv()
                rospy.sleep(1.0)
            self.rate.sleep()
            if self.reset_status != 'moving' and self.trigger:
                self.rate.sleep()
                if self.drop_srv().dropped:  # Check if really grasped
                    self.trigger = False
                    rospy.logerr('[rollout] Grasp failed. Restarting')
                    continue
                else:
                    break
        self.trigger = False

    def slow_open(self):
        print("Opening slowly.")
        for _ in range(30):
            self.move_srv(np.array([-6., -6.]))
            rospy.sleep(0.1)

    def callbackTrigger(self, msg):
        self.reset_status = msg.data
        if not self.trigger and self.reset_status == 'finished':
            self.trigger = True

    def callbackStop(self, msg):
        self.running = msg.data

        return {'success': True, 'message': ''}

    def CallbackRollout(self, req):

        print('[rollout] Rollout request received.')

        actions = np.array(req.actions).reshape(-1, self.action_dim)

        self.S = []
        self.NS = []
        self.A = []
        self.T = []
        self.TP = []
        self.T0 = rospy.get_time()
        self.start_rot = 0.0
        self.start_rot_obj = 0.0

        success = self.run_rollout(actions)

        states = np.array(self.S)
        next_states = np.array(self.NS)
        actions = np.array(self.A)
        target_pos = np.array(self.TP)
        time = np.array(self.T)

        print('[rollout] Recording stopped with %d points.' % len(self.S))

        return {'states': states.reshape((-1,)),
                'actions': actions.reshape((-1,)),
                'next_states': next_states.reshape((-1,)),
                'time': time.reshape((-1,)),
                'success': success,
                'reason': self.reason,
                'target_pos': target_pos.reshape((-1,))
                }

    def CallbackRolloutFile(self, req):

        file_name = req.file

        actions = np.loadtxt(file_name, delimiter=',', dtype=float)[:, :2]
        success = self.run_rollout(actions)

        states = np.array(self.S)

        return {'states': states.reshape((-1,)), 'success': True}

    def callbackObj(self, msg):
        self.obj_pos = np.array(msg.data)

    def callbackObjQ(self, msg):
        self.obj_quat = np.array(msg.data)

    def callbackObjectDrop(self, msg):

        if msg.data:
            self.drop_counter += 1
        else:
            self.drop_counter = 0

        if msg.data and self.drop_counter >= 3:
            self.drop = True
        else:
            self.drop = False


if __name__ == '__main__':
    try:
        rolloutPublisher()
    except rospy.ROSInterruptException:
        pass