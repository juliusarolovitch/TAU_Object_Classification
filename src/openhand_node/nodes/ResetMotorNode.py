#!/usr/bin/python

#Andrew Morgan
#Yale University
#Updated 09/2018

import openhand_node.syncRW_XMHandler as handler
import openhand_node.registerDict as registerDict
import openhand_node.hands as hands
from openhand_node.srv import *
import rospy


class ResetMotorNode():
	hand = None

	#Initialize to have a hand object
	def __init__(self,hand, series):

		self.series = series
		self.hand = hand
		self.sRefs = [0.]*len(self.hand.servos)
		self.syncRW_capabilities = False

		if self.series =='XM' or self.series == 'XH' or self.series == 'X':
			self.XMHandler = handler.syncRW_XMHandler(self.hand)
			self.syncRW_capabilities = True

		for i in xrange(len(self.hand.servos)):
			motor_pos,motor_encoder = self.hand.readMotor(i)
			self.sRefs[i] = motor_pos

		self.is_pos_control = [True] * len(self.hand.servos)
		#initialize service handlers:
		rospy.Service('/openhand_reset_node/move_torque',MoveServos,self.MoveTorqueCallback)
		rospy.Service('/openhand_reset_node/move_servos',MoveServos,self.MoveServosCallback)
		rospy.Service('/openhand_reset_node/torque_servos',TorqueServos,self.TorqueServosCallback)
		rospy.Service('/openhand_reset_node/read_servos',ReadServos,self.ReadServosCallback)
		rospy.Service('/openhand_reset_node/hold_servos',HoldServos,self.HoldServosCallback)
		rospy.Service('/openhand_reset_node/read_current',ReadCurrent,self.ReadCurrentCallback)
		rospy.Service('/openhand_reset_node/read_load',ReadLoad,self.ReadLoadCallback)
		rospy.Service('/openhand_reset_node/read_temperature',ReadTemperature,self.ReadTemperatureCallback)
		rospy.Service('/openhand_reset_node/set_operating_mode',OperatingMode,self.SetOperatingModeCallback)
		rospy.spin()		#blocking

	#This should work in theory, but dynamixels have a mind of their own.
	def HoldServosCallback(self,req):
		self.hand.preventAllLoadErrors(req.offset_scale);
		resp = HoldServosResponse()
		resp.err = 0
		return resp

	def MoveServosCallback(self,req):
		pos_to_move = req.pos
		resp = MoveServosResponse()
		resp.err = 0

		#If only one value was sent for all motors, move them all to that location
		if len(self.hand.servos)>1 and len(pos_to_move)==1:
			pos_to_move = len(self.hand.servos) * [pos_to_move[0]]

		if self.syncRW_capabilities: #These are XM Motors
			self.XMHandler.moveXMmotors(pos_to_move, registerDict.X_Series["ADDR_GOAL_POSITION"], 4)
			self.sRefs = pos_to_move
		else:
			#Move each of the motors as desired for protocol1
			for i in xrange(len(pos_to_move)):
				if i < len(self.hand.servos): #just to be safe
					self.hand.moveMotor(i,pos_to_move[i])
					self.sRefs[i] = pos_to_move[i]	#store the reference value sent through ROS
				else:
					resp.err = 1
		return resp

	def MoveTorqueCallback(self,req):
		torque_to_apply = req.pos
		resp = MoveServosResponse()
		resp.err = 0

		if torque_to_apply: #just to be safe
			self.hand.close_torque(torque_to_apply[0])
		else:
			resp.err = 1

		return resp

	def SetOperatingModeCallback(self, req):
		#If we send in True for position control, we will be in pos_control
		#OW, false will be for Torque control
		resp = OperatingModeResponse()
		resp.err = 0

		#This method really only works for the MX and XM Motors
		is_pos_control = req.pos_control
		servos_to_update = req.servo_indices

		for i in range(len(servos_to_update)):
			if is_pos_control[i] == False:
				self.hand.servos[servos_to_update[i]].enable_torque_mode()
			else: #Then enable position control
				self.hand.servos[servos_to_update[i]].disable_torque_mode()

		return resp


	def TorqueServosCallback(self,req):
		#This typically only works is the finger is already in torque control mode
		resp = TorqueServosResponse()
		resp.err = 0

		torq = req.torq
		servos_to_update = req.servo_indices

		if self.series != 'XM' or self.series != 'MX':
			rospy.logwarn('[ERR] Torque control not implemented for this series of actuator')
			resp.err = 1

		for i in range(len(servos_to_update)):
			self.hand.servos[servos_to_update[i]].apply_torque(torq[i])

		return resp

	def ReadServosCallback(self,req):
		#TODO: Implement sync read for the servos (in all read situations)
		pos = [0.]*len(self.hand.servos)
		ref = self.sRefs
		enc = [0]*len(self.hand.servos)
		for i in xrange(len(self.hand.servos)):
			sp,se = self.hand.readMotor(i)
			pos[i] = sp
			enc[i] = se
		resp = ReadServosResponse()
		resp.pos = pos
		resp.enc = enc
		resp.ref = ref
		return resp

	def ReadCurrentCallback(self,req):
		curr = [0.]*len(self.hand.servos)
		for i in xrange(len(self.hand.servos)):
			curr[i] = self.hand.servos[i].read_current()
		resp = ReadCurrentResponse()
		resp.curr = curr
		return resp

	def ReadLoadCallback(self,req):
		load = [0.]*len(self.hand.servos)
		for i in xrange(len(self.hand.servos)):
			load[i] = self.hand.servos[i].read_load()
		resp = ReadLoadResponse()
		resp.load = list(load)
		return resp

	def ReadTemperatureCallback(self,req):
		temp = [0.]*len(self.hand.servos)
		for i in xrange(len(self.hand.servos)):
			temp[i] = self.hand.servos[i].read_temperature()
		resp = ReadTemperatureResponse()
		resp.temp = temp
		return resp


if __name__=="__main__":

	reset_motor = None
	#initialize ros node elements: (can't read parameters until node is initialized)
	rospy.init_node("ResetMotorNode")

	if rospy.has_param('~reset_openhand_type'):

		reset_model = 'Reset_Motor'
		reset_servo_ids = rospy.get_param('~reset_servo_ids')
		reset_port = rospy.get_param('~reset_servo_port')
		reset_series = rospy.get_param('~reset_servo_type')
		reset_direction = rospy.get_param('~reset_direction')
		reset_motor_offset = rospy.get_param('~reset_motor_offset',[0.])
		reset_abduction_limit = rospy.get_param('~reset_abduction_limit',1)
		if reset_motor_offset == [0.]:
			reset_motor_offset = [0.] * len(reset_servo_ids)

		rospy.loginfo("Initializing "+reset_model+" with servos "+repr(reset_servo_ids)+", port "+reset_port+", Dynamixel model "+reset_series)

		if (reset_model=='Reset_Motor'): # hand selection from yaml file
			if len(reset_servo_ids) != 1:
				rospy.logerr("ERR: expecting 1 servo id, got "+len(reset_servo_ids))
			try:
				reset_motor = hands.Reset_Motor(reset_port,reset_servo_ids[0],reset_series, reset_motor_offset[0])
			except:
				rospy.logerr("ERR: Reset Motor failed to initialize (ResetMotorNode.py)")
	#start node if everything is initialized correctly
	if reset_motor is None:
		rospy.logerr("ERR: Cannot Initialize the reset_motor node (ResetMotorNode.py). Please check to ensure your parameters are correct!")
	else:
		ResetMotorNode(reset_motor, reset_series)

