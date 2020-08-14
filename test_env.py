import random
import numpy as np
import pandas as pd


global s_state
# global vm_state
global num_servers
global s_info
global price_cal

np.set_printoptions(threshold=np.inf)

class Servers(object):
	"""docstring for Server"""
	def __init__(self):
		super(Servers, self).__init__()
		self.n_features = 10
		self.num_task_limit = 10
		
	def price_model(self, time_start, time_end, cpu_usage):
		a = 0.5
		b = 10
		price = [0.12, 0.156, 0.165, 0.117, 0.194, 0.192,
					0.318, 0.266, 0.326, 0.293, 0.388, 0.359,
					0.447, 0.478, 0.513, 0.491, 0.457, 0.506,
					0.640, 0.544, 0.592, 0.486, 0.499, 0.292]

		total_price = 0
		time_start_24 = int(time_start) % 24
		time_end_24 = int(time_end) % 24
		
		# calculate price according different situation
		if cpu_usage < 0.7:
			total_price += ((time_start_24 + 1 - time_start) / 1.0) * (cpu_usage * a) * price[time_start_24]
			total_price += ((time_end - time_end_24) / 1.0) * (cpu_usage * a) * price[time_end_24]
			
			for i in range(int(time_start)+1, int(time_end)):
				total_price += (cpu_usage * a) * price[i%24]

			return total_price
		else:
			total_price += ((time_start_24 + 1 - time_start) / 1.0) * (0.7 * a + b * (cpu_usage - 0.7) * (cpu_usage - 0.7)) * price[time_start_24]
			total_price += ((time_end - time_end_24) / 1.0) * (0.7 * a + b * (cpu_usage - 0.7) * (cpu_usage - 0.7)) * price[time_end_24]
			
			for i in range(int(time_start)+1, int(time_end)):
				total_price += (0.7 * a + b * (cpu_usage - 0.7) * (cpu_usage - 0.7)) * price[i%24]
			
			return total_price

	def server_state(self, server_info):
		global s_state
		global vm_state
		global num_servers
		global s_info
		num_servers = len(server_info)
		s_state = [[[-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0] for j in range(10)] for i in range(len(server_info))]
		s_state = np.array(s_state)
		s_info = server_info
		return s_state


	def server_step(self, task_info, action1): #action (server index, queue index)
		# global price_cal
		global CPU_used
		global RAM_used
		global s_state

		CPU_used = np.zeros((19,1))
		RAM_used = np.zeros((19,1))

		# Current CPU Utilization
		for i in range(len(s_state)):
			for j in range(len(s_state[i])):
				# print(s_state[i][j][2])
				if s_state[i][j][2] != -1:
					CPU_used[i] += s_state[i][j][2]
					RAM_used[i] += s_state[i][j][3]

		Resource_used = np.hstack((CPU_used, RAM_used))

		CPU_used[action1] += task_info[2]


		RAM_used[action1] += task_info[3]

		Resource_used_ = np.hstack((CPU_used, RAM_used))

		if CPU_used[action1] > 0.2 and CPU_used[action1] < 0.8:
			reward_CPU1 = 1
		elif CPU_used[action1] > 1:
			reward_CPU1 = -2
		else:
			reward_CPU1 = -1

		if RAM_used[action1] > 0.2 and RAM_used[action1] < 0.8:
			reward_RAM1 = 1
		elif RAM_used[action1] > 1:
			reward_RAM1 = -2
		else:
			reward_RAM1 = -1

		reward1 = reward_CPU1 + reward_RAM1
		
		return Resource_used, reward1, Resource_used_ 

	def vm_step(self, task_info, action1, action2):#task should be global
		global s_state
		vm_state = s_state[action1]
		vm_state_ = s_state[action1]

		
		for i in range(10): #canshu geshu
			vm_state_[action2][i] = task_info[i] # 


		vm_reward = vm_state_

		# pop the tasks which task_end earlier than time calculated of incoming task
		for i in range(len(vm_reward)):
			if vm_reward[i][6] < task_info[5]:
				for j in range(10):
					vm_reward[i][j] = -1
			else:
				vm_reward[i][8] = task_info[5]

		# here should have a sort finish time 
		vm_reward = vm_reward[np.lexsort(vm_reward[:,:6:].T)]
		# original resource used
		CPU_used = 0
		RAM_used = 0
		for i in range(len(vm_reward)):
			if vm_reward[i][2] != -1:
				CPU_used += vm_reward[i][2]
				RAM_used += vm_reward[i][3] 

		# calculate reward
		reward2 = 0
		for i in range(len(vm_reward)):
			if vm_reward[i][0] == -1:
				continue
			else:
				reward2 += self.price_model(vm_reward[i][8], vm_reward[i][6], (CPU_used / s_info[i][1]))
				CPU_used -= vm_reward[i][2]
				for j in range(i+1, len(vm_reward)):
					vm_reward[j][8] = vm_reward[i][6]
				for j in range(10):
					vm_reward[i][j] = -1
		
		trans_vm_state = vm_state[:, 2:4]
		trans_vm_state_ = vm_state_[:, 2:4]
		trans_vm_state = trans_vm_state.reshape(-1)
		trans_vm_state_ = trans_vm_state_.reshape(-1)


		# for i in range(1, 9):
		# 	trans_vm_state = np.hstack((trans_vm_state, vm_state[i, 2:4]))
		# 	trans_vm_state_ = np.hstack((trans_vm_state, vm_state[i, 2:4]))


		return trans_vm_state, reward2, trans_vm_state_

	def time_step(self, task_info, action1, action2, action3):
		global s_state

		if s_state[action1][action2][0] == -1:
			task_info[5] += action3

			task_info[6] += action3
		else:
			if task_info[5] < s_state[action1][action2][6]:
				task_info[5] = action3 + s_state[action1][action2][6]
				task_info[6] = task_info[5] + task_info[1] - task_info[0]
			else:
				task_info[5] += action3
				task_info[6] += action3
	

		# final updating s_state
		temp_s_state = s_state
		for i in range(len(temp_s_state)):
			for j in range(len(temp_s_state[i])):
				# if finished time is earlier than start time of incoming task
				if temp_s_state[i][j][6] != -1:
					if temp_s_state[i][j][6] < task_info[5]:
						# pop the old tasks
						for k in range(10):
							temp_s_state[i][j][k] = -1
					else:
						temp_s_state[i][j][8] = task_info[5]

		# put the incoming task into server, vm
		for i in range(10):
			temp_s_state[action1][action2][i] = task_info[i]

		# next state, needs to be returned
		s_state_ = temp_s_state

		# this is the state used to calculate total price
		
		reward_state = temp_s_state
		# start of reward state, cut the time_calculated of other performing tasks
		for i in range(len(reward_state)):
			for j in range(len(reward_state[i])):
				if reward_state[i][j][0] != -1:
					reward_state[i][j][8] = task_info[5]


		# pop the tasks which task_end earlier than time calculated of incoming task
		for i in range(len(reward_state)):
			for j in range(len(reward_state[i])):
				if reward_state[i][j][6] < task_info[5]:
					for k in range(10):
						reward_state[i][j][k] = -1
				else:
					reward_state[i][j][8] = task_info[5]

		# calculate the money on the right hand side of red line
		reward3 = 0
		CPU_used = 0
		RAM_used = 0
		for i in range(len(reward_state[action1])):#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!jia ge tiao jian
			if reward_state[action1][i][2] != -1:
				CPU_used += reward_state[action1][i][2]
				RAM_used += reward_state[action1][i][3]
		if CPU_used > s_info[action1][1] or RAM_used > s_info[action1][2] or task_info[6]>task_info[4]:
			reward3 = -1
		else:
			# original resource used
			for i in range(len(reward_state)):
				# here should have a sort finish time shengxu
				reward_state[i] = reward_state[i][np.lexsort(reward_state[i][:,:6:].T)]

			# original resource used on the ith server
				CPU_used = 0
				RAM_used = 0
				for j in range(len(reward_state[i])):
					if reward_state[i][j][2] != -1:
						CPU_used += reward_state[i][j][2]
						RAM_used += reward_state[i][j][3]

			# calculate reward
				for j in range(len(reward_state[i])):
					if reward_state[i][j][0] == -1:
						continue
					else:#
						reward3 += self.price_model(reward_state[i][j][8], reward_state[i][j][6], (CPU_used / s_info[i][1]))
						# print(reward3)
						CPU_used -= reward_state[i][j][2]
						for k in range(j+1, len(reward_state[i][j])):
							reward_state[i][k][8] = reward_state[i][j][6]
						for k in range(10):
							reward_state[i][j][k] = -1


		# total price on the left hand side of red line
		# server_index = action1
		# queue_index = action2
		money_state = s_state
		# calculate resource used
		price_cal = 0
		# if task_info[4] < task_info[6]:
		# 	reward = -2
		# for i in range(len(money_state)):
		# 	CPU_used = 0
		# 	RAM_used = 0
			# for j in range(len(money_state[i])):
			# 	if money_state[i][j][2] != -1:
			# 		CPU_used += money_state[i][j][2]
			# 		RAM_used += money_state[i][j][3]
		# for i in range(len(money_state)):
		# 	temp_server = np.array(money_state[i])
		# 	temp_server = temp_server[np.lexsort(temp_server[:,:6:].T)]
		# 	money_state[i] = temp_server

		money_state[i] = money_state[i][np.lexsort(money_state[i][:,:6:].T)]
		for i in range(len(money_state)):
			CPU_used = 0
			RAM_used = 0
			for j in range(len(money_state[i])):
				if money_state[i][j][2] != -1:
					CPU_used += money_state[i][j][2]
					RAM_used += money_state[i][j][3]
			for j in range(len(money_state[i])):
				if money_state[i][j][6] !=-1:
					if money_state[i][j][6] <= task_info[5]:#
						latest_endtime = money_state[i][j][6]
						price_cal += self.price_model(money_state[i][j][8], money_state[i][j][6], (CPU_used / s_info[i][1]))						
						CPU_used -= money_state[i][j][2]
						RAM_used -= money_state[i][j][3]

						for k in range(j+1, 10):# 9 is size of queue
							money_state[i][k][8] = money_state[i][j][6]
						for k in range(10):
							money_state[i][j][k] = -1
					else:
						price_cal += self.price_model(money_state[i][j][8], task_info[5], (CPU_used / s_info[i][1]))
						for k in range(j,10):# 9 is size of queue
							money_state[i][k][8] = task_info[5]
						break

		trans_s_state = s_state[action1][:, 2:4]
		trans_s_state_ = s_state_[action1][:, 2:4]
		trans_s_state = trans_s_state.reshape(-1)
		trans_s_state_ = trans_s_state_.reshape(-1)
		if reward3 < 0:
			return s_state, trans_s_state, reward3, price_cal, trans_s_state_
		else:
			reward3 = 1 / reward3
			return s_state_, trans_s_state, reward3, price_cal, trans_s_state_

		


