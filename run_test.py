from test_env import Servers
from test_brain import DeepQNetwork
import numpy as np
import pandas as pd

global s_info
global t_info
#global total_price
global s_state
np.set_printoptions(threshold=np.inf)

def read_file():
	global s_info
	global t_info
	# server information
	server_file = pd.read_csv('server.csv')
	server_file.columns=["Col1","Col2","Col3"]
	 
	s_info = server_file[["Col1","Col2","Col3"]]
	s_info = np.array(s_info)

	# task information
	task_file = pd.read_csv('task1.csv')
	task_file.columns=["Col1","Col2","Col3","Col4","Col5"]
	 
	t_info = task_file[["Col1","Col2","Col3","Col4","Col5"]]
	t_info = np.array(t_info)

	# add more data into task information
	t_info = t_info[:,1:]

	a = np.zeros((len(t_info), 6))
	t_info = np.hstack((t_info, a))
	for i in range(len(t_info)):
		t_info[i][4] = t_info[i][1] + np.random.randint(0,10)
		t_info[i][5] = t_info[i][0]
		t_info[i][6] = t_info[i][1]
		t_info[i][8] = t_info[i][0]
	return s_info.shape[0]

def run_server():
	global s_info
	global t_info
	#global total_price
	global s_state
	step = 0 #keep record which step I am in
	for episode in range(100):
		total_price = 0
		# initialize environment
		env = Servers()
		s_state = env.server_state(s_info)


		CPU_used = np.zeros((19,1))
		RAM_used = np.zeros((19,1))
		for i in range(len(s_state)):
			for j in range(len(s_state[i])):
				if s_state[i][j][2] != -1:
					CPU_used[i] += s_state[i][j][2]
					RAM_used[i] += s_state[i][j][3]
		observation1 = np.hstack((CPU_used, RAM_used))
		for i in range(len(t_info)):
			#print(i)
			# if step is less than 20, then choose random server
			if i < 20:
				action1 = np.random.randint(0, 18)
			# else choose from action values
			else:
				CPU_used = np.zeros((19,1))
				RAM_used = np.zeros((19,1))
				for j in range(len(s_state)):

					for k in range(len(s_state[j])):
						if s_state[j][k][2] != -1:
							CPU_used[j] += s_state[j][k][2]
							RAM_used[j] += s_state[j][k][3]
					
				observation1 = np.hstack((CPU_used, RAM_used))
				observation1 = observation1.reshape(-1)
				# print(observation1)
				action1 = RL.choose_server(observation1)


			Resource_used, reward1, Resource_used_= env.server_step(t_info[i], action1)

			RL.store_transition1(Resource_used, action1, reward1, Resource_used_)

			# if step is less than 20, then choose random queue
			if i < 20:
				action2 = np.random.randint(0, 9)
			# else choose from action values
			else:
				observation2 = s_state[action1][:, 2:4]
				observation2 = observation2.reshape(-1)
				action2 = RL.choose_vm(observation2)


			vm_state, reward2, vm_state_ = env.vm_step(t_info[i], action1, action2)
			RL.store_transition2(vm_state, action1, action2, reward2, vm_state_)


			# if step is less than 20, then choose random start time
			if i < 20:
				action3 = np.random.randint(0, 24)
			# else choose from action values
			else:
				observation3 = s_state[action1][:, 2:4]
				observation3 = observation3.reshape(-1)
				action3 = RL.choose_time(observation3)

			s_state_, time_state, reward3, price_cal, time_state_ = env.time_step(t_info[i], action1, action2, action3)
			RL.store_transition3(time_state, action1, action2, action3, reward3, time_state_)

			if (i > 200) and (i % 5 == 0): #start to learn when transition is bigger than 200 and learn every 5 steps
				RL.learn()

			s_state = s_state_
			# print(s_state)
			

			# final price
			total_price += price_cal
		# see if NN is improved
		print(total_price)


if __name__ == "__main__":
	total_price = 0
	num_servers = read_file()
	env = Servers()
	RL = DeepQNetwork(num_servers,
					env.n_features
					)
	run_server()
	RL.plot_cost()
	


