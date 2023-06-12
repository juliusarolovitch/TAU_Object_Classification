#!/usr/bin/python 

'''
Author: Osher Azulay
'''

import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray, Float32MultiArray, String, Bool, Float32, Int16
from std_srvs.srv import Empty, EmptyResponse
from openhand_node.srv import MoveServos, ReadTemperature
from hand_control.srv import TargetAngles, IsDropped, observation, close, ObjOrientation, RegraspObject, TargetPos
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Quaternion, TransformStamped, Vector3, WrenchStamped, \
    Wrench, PoseWithCovariance, Pose
from std_msgs.msg import ColorRGBA, Header
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from apriltag_ros.msg import AprilTagDetectionArray, AprilTagDetection
# from common_msgs_gl.srv import SendDoubleArray, SendBool
import geometry_msgs.msg
import math
import time
import tf

np.set_printoptions(precision=3, suppress=True)


class hand_control():
    finger_initial_offset = np.array([0., 0., 0., 0.])
    finger_opening_position = np.array([0., 0., 0., 0.])
    finger_closing_position = np.array([0., 0., 0., 0.])
    finger_move_offset = np.array([0., 0.01, 0.01, 0.01])  # FIRST IS 0!

    # Gripper properties
    gripper_status = 'open'
    closed_load = np.array(20.)
    gripper_pos = np.array([0., 0., 0., 0.])
    gripper_load = np.array([0., 0., 0., 0.])
    gripper_temperature = np.array([0., 0., 0., 0.])
    target_pos = np.array([0., 0., 0., 0.])

    slip_angle, angle, rot_angle, = 0.0, 0.0, 0.0
    base_pos = [0., 0., 0.]  # w.r.t OptiTrack world
    base_theta, calib_angle = 0.0, 0.0  # w.r.t OptiTrack world
    base_height, obj_height, obj_grasped_height = -1.0e3, -1.0e3, 1.0e3
    obj_pos = [0., 0., 0.]
    obj_rpy = [0., 0., 0.]
    obj_relative_pos = [0., 0., 0.]
    obj_relative_rpy = [0., 0., 0.]
    obj_relative_quat = [0.0, 0.0, 0.0, 0.0]
    obj_pos_pixels = [0, 0]
    start_obj_rpy = [0, 0, 0]
    start_obj_rot = [0, 0, 0, 0]
    R = []
    count = 1
    max_load = 180.0
    max_drop = 5
    drop_counter = 0
    object_grasped = False
    grasped_object_id = 7
    object_map = {0: 'circle', 3: 'hexagon', 4: 'ellipse', 5: 'rectangle', 6: 'square', 9: 'box', 11: 'wire', 10: 'gum',
                  7: 'star'}

    def __init__(self):

        rospy.init_node('hand_control', anonymous=True)

        if rospy.has_param('~finger_initial_offset'):
            self.finger_initial_offset = np.array(rospy.get_param('~finger_initial_offset'))
            self.finger_opening_position = np.array(rospy.get_param('~finger_opening_position'))
            self.finger_closing_position = np.array(rospy.get_param('~finger_closing_position'))
            self.finger_move_offset = np.array(rospy.get_param('~finger_move_offset'))
            self.closed_load = rospy.get_param('~finger_close_load')
            self.max_load = rospy.get_param('~finger_max_load')
            self.sim_step = rospy.get_param('~simulation_step')

        # Gripper related
        rospy.Subscriber('/gripper/pos', Float32MultiArray, self.callbackGripperPos)
        rospy.Subscriber('/gripper/load', Float32MultiArray, self.callbackGripperLoad)
        rospy.Subscriber('/gripper/temperature', Float32MultiArray, self.callbackGripperTemp)

        pub_gripper_status = rospy.Publisher('/hand_control/gripper_status', String, queue_size=10)

        # Object related
        if rospy.get_param('~pose_feedback') == 'optitrack':
            rospy.Subscriber('/marker_tracker/rigid_bodies/object/pose', PoseStamped, self.callbackObjectMarkers)
            rospy.Subscriber('/marker_tracker/rigid_bodies/world/pose', PoseStamped, self.callbackWorldMarkers)
        elif rospy.get_param('~pose_feedback') == 'apriltag':
            rospy.Subscriber('/tag_detections', AprilTagDetectionArray, self.callbackDetection)
            rospy.Subscriber('/object_id', Int16, self.callbackUpdateId)

        pub_drop = rospy.Publisher('/hand_control/drop', Bool, queue_size=10)
        pub_obj_relative_pos = rospy.Publisher('/hand_control/obj_relative_pos', Float32MultiArray, queue_size=10)
        pub_obj_relative_rpy = rospy.Publisher('/hand_control/obj_relative_rpy', Float32MultiArray, queue_size=10)
        pub_obj_relative_quat = rospy.Publisher('/hand_control/obj_relative_quat', Float32MultiArray, queue_size=10)

        pub_obj_pos = rospy.Publisher('/hand_control/obj_pos', Float32MultiArray, queue_size=10)
        pub_obj_rpy = rospy.Publisher('/hand_control/obj_rpy', Float32MultiArray, queue_size=10)
        pub_obj_pos_pixels = rospy.Publisher('/hand_control/obj_pos_pixels', Float32MultiArray, queue_size=10)
        pub_obj_height = rospy.Publisher('/hand_control/object_relative_height', Float32, queue_size=10)

        # Init services
        rospy.Service('/OpenGripper', Empty, self.OpenGripper)
        rospy.Service('/CloseGripper', close, self.CloseGripper)
        rospy.Service('/MoveGripper', TargetAngles, self.MoveGripper)
        rospy.Service('/MoveAdduct', TargetAngles, self.MoveAdduct)
        rospy.Service('/IsObjDropped', IsDropped, self.CheckDroppedSrv)
        rospy.Service('/observation', observation, self.GetObservation)
        rospy.Service('/TargetPos', TargetPos, self.GetTargetPos)

        self.move_servos_srv = rospy.ServiceProxy('/openhand_node/move_servos', MoveServos)
        self.temperature_srv = rospy.ServiceProxy('/openhand_node/read_temperature', ReadTemperature)

        msg = Float32MultiArray()
        msg_ = Float32()
        self.tl = tf.TransformListener()
        self.rate = rospy.Rate(100)

        c = True
        count = 0

        self._check_hand_connection()

        while not rospy.is_shutdown():

            pub_gripper_status.publish(self.gripper_status)

            msg.data = self.obj_pos
            pub_obj_pos.publish(msg)

            msg.data = self.obj_relative_pos
            pub_obj_relative_pos.publish(msg)

            msg.data = self.obj_pos_pixels
            pub_obj_pos_pixels.publish(msg)

            msg.data = self.obj_rpy
            pub_obj_rpy.publish(msg)

            msg.data = self.obj_relative_rpy
            pub_obj_relative_rpy.publish(msg)

            msg.data = self.obj_relative_quat
            pub_obj_relative_quat.publish(msg)

            msg_.data = self.obj_height
            pub_obj_height.publish(msg_)

            if count > 500:
                dr, verbose = self.CheckDropped()
                pub_drop.publish(dr)

            count += 1

            if c and not np.all(self.gripper_load == 0):
                self.slow_open()
                self.moveGripper(self.finger_opening_position)
                c = False

            self.rate.sleep()

    '''
    Subscribers
    '''

    def _check_hand_connection(self):
        self.gripper_motor_state = None
        rospy.logwarn(
            "Waiting for gripper/pos to be READY...")
        while self.gripper_motor_state is None and not rospy.is_shutdown():
            try:
                self.gripper_motor_state = rospy.wait_for_message(
                    "gripper/pos", Float32MultiArray, timeout=5.0)
                rospy.logwarn(
                    "Current gripper/pos READY=>")

            except:
                rospy.logerr(
                    "Current gripper/pos not ready yet, retrying")

        return self.gripper_motor_state

    def callbackGripperPos(self, msg):
        self.gripper_pos = np.array(msg.data)

    def callbackGripperLoad(self, msg):
        self.gripper_load = np.array(msg.data)

    def callbackGripperTemp(self, msg):
        self.gripper_temperature = np.array(msg.data)

    def callbackUpdateId(self, msg):
        self.grasped_object_id = msg.data

    def callbackDetection(self, msg):

        '''
        Relative Pose
        '''

        detected_id = []
        detection_array = msg.detections
        for detect in detection_array:
            detected_id.append(detect.id[0])

        if self.grasped_object_id not in detected_id:
            self.obj_pos = np.array([np.nan, np.nan, np.nan])
            self.obj_rpy = np.array([np.nan, np.nan, np.nan])
            self.obj_relative_pos = np.array([np.nan, np.nan, np.nan])
            self.obj_relative_rpy = np.array([np.nan, np.nan, np.nan])
            self.obj_relative_quat = np.array([np.nan, np.nan, np.nan, np.nan])
            self.obj_pos_pixels = np.array([np.nan, np.nan])
            self.rot_angle = None
            self.obj_height = None
            self.slip_angle = None
        else:
            for detect in detection_array:

                '''
                Object pose with respect to camera frame
                '''
                pose = detect.pose.pose.pose  # PoseWithCovarianceStamped -> PoseWithCovariance -> Pose(position, orientation)
                if detect.id[0] == self.grasped_object_id:
                    self.obj_relative_pos = np.array([pose.position.x, pose.position.y, pose.position.z])
                    self.obj_relative_quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z,
                                              pose.orientation.w]
                    roll, pitch, yaw = euler_from_quaternion(self.obj_relative_quat)
                    self.obj_relative_rpy = np.array([roll, pitch, yaw])
                    self.obj_height = self.obj_relative_pos[-1]
                    self.slip_angle = abs(pitch)
                    self.rot_angle = yaw
                    self.obj_pos_pixels = np.array([detect.pixels[0], detect.pixels[1]])

                    ###############

                    #
                    # self.obj_rpy[0] = self.obj_rpy[0] - self.start_roll + self.start_roll_obj
                    # if self.obj_rpy[0] > np.pi: self.obj_rpy[0] -= 2. * np.pi
                    # if self.obj_rpy[0] < -np.pi: self.obj_rpy[0] += 2. * np.pi
                    ############
                    '''
                    Object pose with respect to world frame
                    '''
                    ps = PoseStamped()
                    ps.header = detect.pose.header
                    ps.header.stamp = rospy.Time(0)
                    ps.pose = detect.pose.pose.pose
                    try:
                        # p = self.tl.transformPose('world', ps)  # from camera to world
                        # obj_pos = p.pose
                        # self.obj_pos = np.array([obj_pos.position.x, obj_pos.position.y, obj_pos.position.z])
                        # roll, pitch, yaw = euler_from_quaternion(
                        #     [obj_pos.orientation.x, obj_pos.orientation.y, obj_pos.orientation.z, obj_pos.orientation.w])
                        trans, rot, mat = self.tf_trans('world', self.object_map[self.grasped_object_id])
                        T_center = np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0],
                                             [0, 0, 1, -0.06],
                                             [0, 0, 0, 1]])
                        try:
                            T0G = np.matmul(mat, T_center)
                            self.obj_pos = T0G[:3, -1]  # trans
                            roll, pitch, yaw = euler_from_quaternion(rot)
                            self.obj_rpy = np.array([roll, pitch, yaw])
                            if not np.sum(self.start_obj_rpy):
                                self.start_obj_rpy = self.obj_rpy
                            # self.obj_height = trans[2]
                        except:
                            self.obj_pos = np.array([np.nan, np.nan, np.nan])
                            self.obj_rpy = np.array([np.nan, np.nan, np.nan])
                            self.obj_height = None
                        # rospy.logerr(np.degrees(self.obj_rpy))

                    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                        rospy.logerr("TF start to end not ready YET...")
                        pass

        # rospy.logerr("pos: " +str(self.obj_pos) + "   rpy:  " + str(np.degrees(self.obj_rpy)))

    def tf_trans(self, target_frame, source_frame):
        try:
            # listen to transform, from source to target. if source is 0 and target is 1 than A_0^1
            (trans, rot) = self.tl.lookupTransform(target_frame, source_frame, rospy.Time(0))
            mat = self.tl.fromTranslationRotation(trans, rot)
            return trans, rot, mat
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return 'couldnt find mat', None, None

    def callbackWorldMarkers(self, msg):
        try:
            quaternion = (
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w)
            roll, pitch, yaw = euler_from_quaternion(quaternion)
            self.base_theta = np.array(yaw)
            self.base_pos = np.array([msg.pose.position.x, msg.pose.position.y])
            self.base_height = msg.pose.position.z
        except:
            self.base_pos = np.array([np.nan, np.nan])
            self.base_theta = np.nan
            self.base_height = np.nan

    def callbackObjectMarkers(self, msg):
        # The direction of the transform returned will be from the target_frame to the source_frame.
        # Which if applied to data, will transform data in the source_frame into the target_frame
        #                                       target   source
        (trans, rot) = self.tl.lookupTransform("world", "object", rospy.Time(0))
        try:
            quaternion = rot
            roll, pitch, yaw = euler_from_quaternion(quaternion)
            self.angle = np.array(np.arctan2(trans[0], trans[1]))
            self.rot_angle = yaw
            self.obj_pos = np.array([trans[1], trans[0], trans[2]])
            self.obj_height = trans[2]
            self.slip_angle = math.sqrt(roll ** 2 + pitch ** 2) * 180 / math.pi
        except:
            self.obj_pos = np.array([np.nan, np.nan, np.nan])
            self.angle = np.nan
            self.rot_angle = np.nan
            self.obj_height = np.nan

    '''
    Services
    '''

    def OpenGripper(self, msg):

        self.slow_open()
        self.gripper_status = 'open'
        self.moveGripper(self.finger_opening_position, open=True)

        return EmptyResponse()

    def slow_open(self):

        for _ in range(60):
            f = 2  # 2 # 50.0
            inc = np.array([0, -0.001, -0.001, -0.001])
            t = rospy.get_time()
            while (rospy.get_time() - t < 0.1) and np.any(self.gripper_pos[1:] > self.finger_opening_position[1]):
                if np.all(self.gripper_pos[1:] < 0.01):
                    break
                self.target_pos += inc * 1.0 / f
                self.moveGripper(self.target_pos)

    def CloseGripper(self, msg):
        # GRASP OBJECT
        # CHECK OVERHEAT
        count = 0
        if np.any(self.gripper_temperature > 52.):
            rospy.logdebug('[hand_control] Actuators overheated, taking a break...')
            while 1:
                if np.all(self.gripper_temperature < 60.):
                    break
                self.rate.sleep()

        closed_load = self.closed_load

        self.object_grasped = False
        for i in range(400):
            # CHECK LOAD
            if np.any(np.abs(self.gripper_load) > closed_load):
                count += 1
            if count > 5:
                # rospy.loginfo('[hand_control] Object grasped.')
                self.gripper_status = 'closed'
                break
            # CHECK GRASP ANGLE LIMITS
            desired = self.gripper_pos + np.array(
                [a for a in self.finger_move_offset])  # slow close [a for a in self.finger_move_offset]
            if np.any(desired > 0.7):
                rospy.logdebug('[hand_control] Desired angles out of bounds to grasp object.')
                break
            self.moveGripper(desired)
            rospy.sleep(0.05)

        self.rate.sleep()

        # Verify grasp based on height - not useful if camera cannot see
        # print('[hand_control] Object height relative to gripper : '+ str(self.obj_height))
        if self.obj_height is not None:
            if abs(self.obj_height) < 0.1:
                self.object_grasped = True
                self.obj_grasped_height = self.obj_height

        # print('[hand_control] Gripper actuator angles: ' + str(self.gripper_pos))
        self.rate.sleep()
        self.target_pos = self.gripper_pos

        return {'success': self.object_grasped}

    def MoveGripper(self, msg):
        '''
            This function should accept a vector of normalized increments to the current angles:
            msg.angles = [dq1, dq2, dq3], where dq1 and dq2 can be equal to 0 (no move), 1,-1
            (increase or decrease angles by finger_move_offset)
        '''
        self.target_pos = self.gripper_pos

        inc = np.array(msg.angles)
        inc_angles = np.multiply(self.finger_move_offset, inc)

        self.target_pos += inc_angles

        suc = self.moveGripper(self.target_pos)

        self.wait_for_joints_to_get_there(self.target_pos)

        return {'success': suc}

    def MoveAdduct(self, msg):
        '''
        This function should accept a vector of normalized increments to the current angles:
        msg.angles = [dq1, dq2, dq3], where dq1 and dq2 can be equal to 0 (no move), 1,-1
         (increase or decrease angles by finger_move_offset)
        '''
        f = 2
        inc = np.array(msg.angles)
        inc_angles = np.multiply(np.array([0.005, 0.0, 0.0, 0.0]), inc)

        t = rospy.get_time()
        while rospy.get_time() - t < 0.1:
            self.target_pos += inc_angles * 1.0 / f
            suc = self.moveGripperAdd(self.target_pos)

        return {'success': suc}

    def moveGripper(self, angles, open=False):

        if not open:
            if np.any(angles[1:] > 0.9) or np.any(angles[1:] < -0.02):
                rospy.logerr('[hand_control] Move Failed. Desired angles out of bounds.')
                return False

            if np.any(abs(self.gripper_load) > self.max_load):
                rospy.logerr('[hand_control] Move failed. Pre-overload.')
                return False

        self.move_servos_srv.call(angles)
        # rospy.sleep(0.05)
        return True

    def moveGripperAdd(self, angles, open=False):

        if not open:
            if np.any(angles > 0.9) or np.any(angles < -0.02):
                rospy.logerr('[hand_control] Move Failed. Desired angles out of bounds.')
                return False

            if np.any(abs(self.gripper_load) > self.max_load):
                rospy.logerr('[hand_control] Move failed. Pre-overload.')
                return False

        self.move_servos_srv.call(angles)
        # rospy.sleep(0.05)
        return True

    def CheckDropped(self):

        try:
            if np.any(self.gripper_pos > 0.9) or np.any(self.gripper_pos < -0.05):
                verbose = '[hand_control] Desired angles out of bounds -  assume dropped.'
                return True, verbose
        except:
            print('[hand_control] Error with gripper_pos.')

        # # Check load
        # if np.any(abs(self.gripper_load) > self.max_load):
        #     verbose = '[hand_control] Pre-overload.'
        #     return True, verbose

        if self.obj_height < 0.02 or self.obj_height is None:
            verbose = '[hand_control] Object is too close. height: ' + str(self.obj_height)
            return True, verbose

        if self.obj_height > 0.1 or self.obj_height is None:
            verbose = '[hand_control] Object is dropped. height: ' + str(self.obj_height)
            return True, verbose

        relative_rpy = np.degrees(self.obj_rpy - self.start_obj_rpy)
        if np.any(np.abs(relative_rpy[:2])) > 20 or np.isnan(np.sum(relative_rpy)):
            verbose = '[hand_control] Object slipped. slip_angle: ' + str(relative_rpy)
            rospy.logerr('[hand_control] Object slipped. slip_angle: ' + str(relative_rpy))
            return True, verbose

        return False, ''

    def CheckDroppedSrv(self, msg):

        dr, verbose = self.CheckDropped()

        if len(verbose) > 0: rospy.logdebug(verbose)

        return {'dropped': dr}

    def GetObservation(self, msg):

        obs = np.concatenate((self.obj_relative_pos,
                              self.obj_relative_quat,
                              self.gripper_pos[1:],
                              self.gripper_load[1:],
                              self.target_pos[1:]))
        return {'state': obs}

    def GetTargetPos(self, msg):
        return {'angle': self.target_pos}

    def wait_for_joints_to_get_there(self, desired_pos_array, error=0.2, timeout=3.0):

        time_waiting = 0.0
        frequency = 10.0
        are_equal = False
        is_timeout = False
        rate = rospy.Rate(frequency)
        rospy.logwarn("Waiting for joint to get to the position")
        while not are_equal and not is_timeout and not rospy.is_shutdown():

            current_pos = self.gripper_pos

            are_equal = np.allclose(a=current_pos,
                                    b=desired_pos_array,
                                    atol=error)

            rospy.logdebug("are_equal=" + str(are_equal))
            rospy.logdebug(str(desired_pos_array))
            rospy.logdebug(str(current_pos))

            rate.sleep()
            if timeout == 0.0:
                # We wait what it takes
                time_waiting += 0.0
            else:
                time_waiting += 1.0 / frequency
            is_timeout = time_waiting > timeout

        rospy.logwarn(
            "Joints are in the desired position with an erro of " + str(error))


if __name__ == '__main__':

    try:
        hand_control()
    except rospy.ROSInterruptException:
        pass