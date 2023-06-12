#!/usr/bin/env python
import pprint

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Float32MultiArray, Bool
from std_srvs.srv import Empty, EmptyResponse, SetBool
from rollout.srv import rolloutReqMod, rolloutReqFile, observation, IsDropped, TargetAngles, gets
from hand_control.srv import RegraspObject, close, TargetPos
import numpy as np

pp = pprint.PrettyPrinter(indent=4)

prefix = ''
joints = ['base_to_finger_1_1', 'base_to_finger_2_1',
          'finger_1_1_to_finger_1_2', 'finger_1_2_to_finger_1_3',
          'finger_2_1_to_finger_2_2','finger_2_2_to_finger_2_3',
          'base_to_finger_3_2', 'finger_3_2_to_finger_3_3']

joints_list = [prefix+'::'+joint for joint in joints]
node_name = prefix + 'joint_publisher'


def main():
    CustomPublisher()
    rospy.spin()


class CustomPublisher:
    rospy.init_node(node_name)

    Hz = 100
    rate = rospy.Rate(Hz)
    gripper_pos = np.array([0., 0., 0., 0.])
    gripper_vel = np.array([0., 0., 0., 0.])
    gripper_load = np.array([0., 0., 0., 0.])

    def __init__(self):

        pub = rospy.Publisher('joint_states', JointState, queue_size=1)
        rospy.Subscriber('/gripper/pos', Float32MultiArray, self.callbackGripperPos)
        rospy.Subscriber('/gripper/load', Float32MultiArray, self.callbackGripperLoad)

        while not rospy.is_shutdown():
            state = JointState()
            try:
                ##### base
                mul = 6
                state.name.append('base_to_finger_1_1')
                state.position.append(self.gripper_pos[0] + 0.6)
                state.velocity.append(self.gripper_vel[0])
                state.effort.append(self.gripper_load[0])
                state.name.append('base_to_finger_2_1')
                state.position.append(-(self.gripper_pos[0] + 0.6))
                state.velocity.append(self.gripper_vel[0])
                state.effort.append(self.gripper_load[0])

                state.name.append('finger_1_1_to_finger_1_2')
                state.position.append(min(mul*self.gripper_pos[1] + 0.4,1.5))
                state.velocity.append(self.gripper_vel[1])
                state.effort.append(self.gripper_load[1])
                state.name.append('finger_1_2_to_finger_1_3')
                state.position.append(2*self.gripper_pos[1]*0.5)
                state.velocity.append(self.gripper_vel[1]*0.5)
                state.effort.append(self.gripper_load[1]*0.5)

                state.name.append('finger_2_1_to_finger_2_2')
                state.position.append(min(mul*self.gripper_pos[2] + 0.4,1.5))
                state.velocity.append(self.gripper_vel[2])
                state.effort.append(self.gripper_load[2])
                state.name.append('finger_2_2_to_finger_2_3')
                state.position.append(2*self.gripper_pos[2]*0.5)
                state.velocity.append(self.gripper_vel[2]*0.5)
                state.effort.append(self.gripper_load[2]*0.5)

                state.name.append('base_to_finger_3_2')
                state.position.append(min(mul*self.gripper_pos[3] + 0.4,1.5))
                state.velocity.append(self.gripper_vel[3])
                state.effort.append(self.gripper_load[3])
                state.name.append('finger_3_2_to_finger_3_3')
                state.position.append(2*self.gripper_pos[3]*0.5)
                state.velocity.append(self.gripper_vel[3]*0.5)
                state.effort.append(self.gripper_load[3]*0.5)

            except Exception as e:
                rospy.logerr("Exception in Joint State Publisher.")
                print(type(e))
                print(e)
                return

            pub.publish(state)
            self.rate.sleep()
    
    def callbackGripperPos(self, msg):
        self.gripper_vel = (np.array(msg.data) - self.gripper_pos) * self.Hz
        self.gripper_pos = np.array(msg.data)

    def callbackGripperLoad(self, msg):
        self.gripper_load = np.array(msg.data)



if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass