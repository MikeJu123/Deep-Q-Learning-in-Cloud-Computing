import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

global CPU_used
global RAM_used
# global transition

# Deep Q Network off-policy
class DeepQNetwork: 
	def __init__( 
			self, #self is an instance in class, is the address of the current object, initialize anything by "self.*** = ***"
			n_actions,
			n_features
	): #"__init__" is like constructor in C++

		self.n_actions1 = n_actions
		self.n_actions2 = 10
		self.n_actions3 = 24 # start time can not be delay over 24 hrs
		self.n_features1 = n_actions * 2
		self.n_features2 = 10 * 2 
		self.n_features3 = 10 * 2 #
		self.lr = 0.01
		self.gamma = 0.9
		self.epsilon_max = 0.9
		self.replace_target_iter = 300 #every replace_target_iter step, update target value
		self.memory_size = 500 #memory
		self.batch_size = 32
		self.epsilon_increment = None
		self.epsilon = 0 if None is not None else self.epsilon_max

		# total learning step
		self.learn_step_counter = 0

		# initialize zero memory [s, a, r, s_], 200*6
		self.memory1 = np.zeros((self.memory_size, self.n_features1 * 2 + 2)) #38 * 2 + 1 action + 1 reward
		self.memory2 = np.zeros((self.memory_size, self.n_features2 * 2 + 3)) 
		self.memory3 = np.zeros((self.memory_size, self.n_features3 * 2 + 4))

		# consist of [target_net, evaluate_net]
		self._build_net1()
		#get_collection(): get all the elements in the specified name, creaet a list and return it
		#tf.GraphKeys.GLOBAL_VARIABLE()
		t_params1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net1')
		e_params1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net1')

		self._build_net2()
		t_params2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net2')
		e_params2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net2')

		self._build_net3()
		t_params3 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net3')
		e_params3 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net3')

		with tf.variable_scope('hard_replacement1'):
			self.target_replace_op1 = [tf.assign(t1, e1) for t1, e1 in zip(t_params1, e_params1)]

		with tf.variable_scope('hard_replacement2'):
			self.target_replace_op2 = [tf.assign(t2, e2) for t2, e2 in zip(t_params2, e_params2)]

		with tf.variable_scope('hard_replacement3'):
			self.target_replace_op3 = [tf.assign(t3, e3) for t3, e3 in zip(t_params3, e_params3)]

		self.sess = tf.Session()

		# if output_graph:
		# 	# $ tensorboard --logdir=logs
		# 	tf.summary.FileWriter("logs/", self.sess.graph)

		self.sess.run(tf.global_variables_initializer())
		self.cost_table1 = []
		self.cost_table2 = []
		self.cost_table3 = []

	# 
	def _build_net1(self):
		# ------------------ all inputs ------------------------
		self.s1 = tf.placeholder(tf.float32, [None, self.n_actions1*2], name='s1')  # input State, n_features: number of feature
		self.s1_ = tf.placeholder(tf.float32, [None, self.n_actions1*2], name='s1_')  # input Next State
		self.r1 = tf.placeholder(tf.float32, [None, ], name='r1')  # input Reward
		self.a1 = tf.placeholder(tf.int32, [None, ], name='a1')  # input Action

		w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

		#the following two networks has the same structure
		# ------------------ build evaluate_net1 ------------------
		with tf.variable_scope('eval_net1'):
			#the first layer has 20 units, state as inputs
			e1 = tf.layers.dense(self.s1, 10, tf.nn.relu, kernel_initializer=w_initializer,
								 bias_initializer=b_initializer, name='e1')
			middle_e1 = tf.layers.dense(e1, 10, tf.nn.relu, kernel_initializer=w_initializer,
								 bias_initializer=b_initializer, name='middle_e1')
			#the second layer has n_action units, the result of e1 as inputs
			self.q_eval1 = tf.layers.dense(middle_e1, self.n_actions1, kernel_initializer=w_initializer,
										  bias_initializer=b_initializer, name='q1')
			

		# ------------------ build target_net ------------------
		with tf.variable_scope('target_net1'):
			t1 = tf.layers.dense(self.s1_, 10, tf.nn.relu, kernel_initializer=w_initializer,
								 bias_initializer=b_initializer, name='t1')
			middle_t1 = tf.layers.dense(t1, 10, tf.nn.relu, kernel_initializer=w_initializer,
								 bias_initializer=b_initializer, name='middle_t1')
			self.q_next1 = tf.layers.dense(middle_t1, self.n_actions1, kernel_initializer=w_initializer,
										  bias_initializer=b_initializer, name='qn1')

		with tf.variable_scope('q_target1'):
			q_target1 = self.r1 + self.gamma * tf.reduce_max(self.q_next1, axis=1, name='Qmax1_s_')    # shape=(None, )
			self.q_target1 = tf.stop_gradient(q_target1)
		with tf.variable_scope('q_eval1'):
			a_indices1 = tf.stack([tf.range(tf.shape(self.a1)[0], dtype=tf.int32), self.a1], axis=1)
			self.q_eval_wrt_a1 = tf.gather_nd(params=self.q_eval1, indices=a_indices1)    # shape=(None, )
		with tf.variable_scope('loss1'):
			self.loss1 = tf.reduce_mean(tf.squared_difference(self.q_target1, self.q_eval_wrt_a1, name='TD_error1'))
		#train operation:
		with tf.variable_scope('train1'):
			self._train_op1 = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss1)


	def _build_net2(self):
		# ------------------ all inputs ------------------------
		self.s2 = tf.placeholder(tf.float32, [None, 20], name='s2')  # input State, n_features: number of feature
		self.s2_ = tf.placeholder(tf.float32, [None, 20], name='s2_')  # input Next State
		self.r2 = tf.placeholder(tf.float32, [None, ], name='r2')  # input Reward
		self.a2 = tf.placeholder(tf.int32, [None, ], name='a2')  # input Action

		w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
		# print(b_initializer)

		#the following two networks has the same structure
		# ------------------ build evaluate_net ------------------
		with tf.variable_scope('eval_net2'):
			#the first layer has 20 units, state as inputs
			e2 = tf.layers.dense(self.s2, 10, tf.nn.relu, kernel_initializer=w_initializer,
								 bias_initializer=b_initializer, name='e2')
			middle_e2 = tf.layers.dense(e2, 10, tf.nn.relu, kernel_initializer=w_initializer,
								 bias_initializer=b_initializer, name='middle_e2')
			#the second layer has n_action units, the result of e1 as inputs
			self.q_eval2 = tf.layers.dense(middle_e2, self.n_actions2, kernel_initializer=w_initializer,
										  bias_initializer=b_initializer, name='q2')

		# ------------------ build target_net ------------------
		with tf.variable_scope('target_net2'):
			t2 = tf.layers.dense(self.s2_, 10, tf.nn.relu, kernel_initializer=w_initializer,
								 bias_initializer=b_initializer, name='t2')
			middle_t2 = tf.layers.dense(t2, 10, tf.nn.relu, kernel_initializer=w_initializer,
								 bias_initializer=b_initializer, name='middle_t2')
			self.q_next2 = tf.layers.dense(middle_t2, self.n_actions2, kernel_initializer=w_initializer,
										  bias_initializer=b_initializer, name='qn2')


		with tf.variable_scope('q_target2'):
			q_target2 = self.r2 + self.gamma * tf.reduce_max(self.q_next2, axis=1, name='Qmax2_s_')    # shape=(None, )
			self.q_target2 = tf.stop_gradient(q_target2)
		with tf.variable_scope('q_eval2'):
			# temp = tf.unstack(self.a2, axis = 1)

			a_indices2 = tf.stack([tf.range(tf.shape(self.a2)[0], dtype=tf.int32), self.a2], axis=1)
			#a_indices2 = tf.stack([tf.range(tf.shape(self.a2)[0], dtype=tf.int32), temp[0], temp[1]], axis=1)
			self.q_eval_wrt_a2 = tf.gather_nd(params=self.q_eval2, indices=a_indices2)    # shape=(None, )
		with tf.variable_scope('loss2'):
			self.loss2 = tf.reduce_mean(tf.squared_difference(self.q_target2, self.q_eval_wrt_a2, name='TD_error2'))
		#train operation:
		with tf.variable_scope('train2'):
			self._train_op2 = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss2)



	def _build_net3(self):
		# ------------------ all inputs ------------------------
		self.s3 = tf.placeholder(tf.float32, [None, 20], name='s3')  # input State, n_features: number of feature
		self.s3_ = tf.placeholder(tf.float32, [None, 20], name='s3_')  # input Next State
		self.r3 = tf.placeholder(tf.float32, [None, ], name='r3')  # input Reward
		self.a3 = tf.placeholder(tf.int32, [None, ], name='a3')  # input Action

		w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

		#the following two networks has the same structure
		# ------------------ build evaluate_net ------------------
		with tf.variable_scope('eval_net3'):
			#the first layer has 20 units, state as inputs
			e3 = tf.layers.dense(self.s3, 10, tf.nn.relu, kernel_initializer=w_initializer,
								 bias_initializer=b_initializer, name='e3')
			middle_e3 = tf.layers.dense(e3, 10, tf.nn.relu, kernel_initializer=w_initializer,
								 bias_initializer=b_initializer, name='middle_e3')
			#the second layer has n_action units, the result of e1 as inputs
			self.q_eval3 = tf.layers.dense(middle_e3, self.n_actions3, kernel_initializer=w_initializer,
										  bias_initializer=b_initializer, name='q3')

		# ------------------ build target_net ------------------
		with tf.variable_scope('target_net3'):
			t3 = tf.layers.dense(self.s3_, 10, tf.nn.relu, kernel_initializer=w_initializer,
								 bias_initializer=b_initializer, name='t3')
			middle_t3 = tf.layers.dense(t3, 10, tf.nn.relu, kernel_initializer=w_initializer,
								 bias_initializer=b_initializer, name='middle_t3')
			self.q_next3 = tf.layers.dense(middle_t3, self.n_actions3, kernel_initializer=w_initializer,
										  bias_initializer=b_initializer, name='qn3')


		with tf.variable_scope('q_target3'):
			q_target3 = self.r3 + self.gamma * tf.reduce_max(self.q_next3, axis=1, name='Qmax3_s_')    # shape=(None, )
			self.q_target3 = tf.stop_gradient(q_target3)
		with tf.variable_scope('q_eval3'):
			a_indices3 = tf.stack([tf.range(tf.shape(self.a3)[0], dtype=tf.int32), self.a3], axis=1)
			self.q_eval_wrt_a3 = tf.gather_nd(params=self.q_eval3, indices=a_indices3)    # shape=(None, )
		with tf.variable_scope('loss3'):
			self.loss3 = tf.reduce_mean(tf.squared_difference(self.q_target3, self.q_eval_wrt_a3, name='TD_error2'))
		#train operation:
		with tf.variable_scope('train3'):
			self._train_op3 = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss3)



	def store_transition1(self, s1, a1, r1, s1_):
	#if "self" doesn't have this reference, then create one
		
		s1 = s1.reshape(-1)
		s1_ = s1_.reshape(-1)
		# print(s1)
		# print(a1)
		# print(r1)
		# print(s1_)

		if hasattr(self, 'memory_counter') is False:
			self.memory_counter = 0

		# np.hstack(): horizontally stack every array input
		transition1 = np.hstack((s1, a1, r1, s1_))
		# print(transition1)

		#replace the old values every "memory_size"
		#because the size of memery is just this big, we can eliminate the older values
		index = self.memory_counter % self.memory_size
		#at the memory_counter row, put this line of value into the memory, updating the old with the new
		self.memory1[index, :] = transition1

		# print(self.memory)
		# self.memory_counter += 1

	def store_transition2(self, s2, a1, a2, r2, s2_):
		if hasattr(self, 'memory_counter') is False:
			self.memory_counter = 0

		# np.hstack(): horizontally stack every array input
		transition2 = np.hstack((s2, a1, a2, r2, s2_))
		# print(transition2)

		#replace the old values every "memory_size"
		#because the size of memery is just this big, we can eliminate the older values
		index = self.memory_counter % self.memory_size
		#at the memory_counter row, put this line of value into the memory, updating the old with the new
		self.memory2[index, :] = transition2

		self.memory_counter += 1

	def store_transition3(self, s3, a1, a2, a3, r3, s3_):

		if hasattr(self, 'memory_counter') is False:
			self.memory_counter = 0#shi fou xu yao memory count3

		# np.hstack(): horizontally stack every array input
		transition3 = np.hstack((s3, a1, a2, a3, r3, s3_))
		# print(transition3)
		#replace the old values every "memory_size"
		#because the size of memery is just this big, we can eliminate the older values
		index = self.memory_counter % self.memory_size
		#at the memory_counter row, put this line of value into the memory, updating the old with the new
		self.memory3[index, :] = transition3
		# print(self.memory)
		self.memory_counter += 1


	def choose_server(self, observation1): #observation):
		# to have batch dimension when feed into tf placeholder
		# at the begining, observation is a one dimension array, to calculate it, transform it into two dimension array
		#############################################################################################################

		##############################################################################################################
		# observation = observation[np.newaxis, :]
		# print(observation1.shape)
		# print(observation1)
		observation1 = np.expand_dims(observation1, axis=0)
		if np.random.uniform() < self.epsilon:  #90% of chance to pick the existed biggest value
			# forward feed the observation and get q value for every actions
			actions_value1 = self.sess.run(self.q_eval1, feed_dict={self.s1: observation1})
			action1 = np.argmax(actions_value1)
			# print(actions_value1)
			# print(action1)
		else:                                   #10% of chance to pick the random value
			action1 = np.random.randint(0, self.n_actions1)
			# print(action)
		return action1

	def choose_vm(self, observation2): #observation):
		# to have batch dimension when feed into tf placeholder
		# at the begining, observation is a one dimension array, to calculate it, transform it into two dimension array
		#############################################################################################################

		##############################################################################################################
		# print(observation)
		observation2 = np.expand_dims(observation2, axis=0)
		if np.random.uniform() < self.epsilon:  #90% of chance to pick the existed biggest value
			actions_value2 = self.sess.run(self.q_eval2, feed_dict={self.s2: observation2})
			action2 = np.argmax(actions_value2)
			# print(action)
		else:
			action2 = np.random.randint(0, self.n_actions2)
			# print(action)
		return action2


	def choose_time(self, observation3): #
		# to have batch dimension when feed into tf placeholder
		# at the begining, observation is a one dimension array, to calculate it, transform it into two dimension array
		############################################################################################################
		observation3 = np.expand_dims(observation3, axis=0)
		if np.random.uniform() < self.epsilon:  #90% of chance to pick the existed biggest value
			actions_value3 = self.sess.run(self.q_eval3, feed_dict={self.s3: observation3})
			action3 = np.argmax(actions_value3)
			# print(action)
		else:
			action3 = np.random.randint(0, self.n_actions3)
			# print(action)
		return action3

	# 
	def learn(self):
		# check to replace target parameters
		if self.learn_step_counter % self.replace_target_iter == 0:
			self.sess.run(self.target_replace_op1)
			self.sess.run(self.target_replace_op2)
			self.sess.run(self.target_replace_op3)
			print('\ntarget_params_replaced\n')

		# sample batch memory from all memory
		if self.memory_counter > self.memory_size:
			sample_index = np.random.choice(self.memory_size, size=self.batch_size)
		else:
			sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
		batch_memory1 = self.memory1[sample_index, :]
		batch_memory2 = self.memory2[sample_index, :]
		batch_memory3 = self.memory3[sample_index, :]
		# print(batch_memory)

		_, cost1 = self.sess.run(
			[self._train_op1, self.loss1],
			feed_dict={
				# self.s:  batch_memory[0],
				# self.a:  batch_memory[1],
				# self.r:  batch_memory[2],
				# self.s_:  batch_memory[3],
				self.s1: batch_memory1[:, :self.n_features1],
				self.a1: batch_memory1[:, self.n_features1],
				self.r1: batch_memory1[:, self.n_features1 + 1],
				self.s1_: batch_memory1[:, -self.n_features1:],
			})

		_, cost2 = self.sess.run(
			[self._train_op2, self.loss2],
			feed_dict={
				self.s2: batch_memory2[:, :self.n_features2],
				self.a2: batch_memory2[:, self.n_features2 + 1],
				self.r2: batch_memory2[:, self.n_features2 + 2],
				self.s2_: batch_memory2[:, -self.n_features2:],
			})

		_, cost3 = self.sess.run(
			[self._train_op3, self.loss3],
			feed_dict={
				self.s3: batch_memory3[:, :self.n_features3],
				self.a3: batch_memory3[:, self.n_features3 + 2],
				self.r3: batch_memory3[:, self.n_features3 + 3],
				self.s3_: batch_memory3[:, -self.n_features3:],
			})

		self.cost_table1.append(cost1)
		self.cost_table2.append(cost2)
		self.cost_table3.append(cost3)


		# increasing epsilon
		self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
		self.learn_step_counter += 1
		#print(self.learn_step_counter)

	def plot_cost(self):
			import matplotlib.pyplot as plt
			plt.figure()
			plt.subplot(1,3,1)
			plt.plot(np.arange(len(self.cost_table1)), self.cost_table1)
			plt.subplot(1,3,2)
			plt.plot(np.arange(len(self.cost_table2)), self.cost_table2)
			plt.subplot(1,3,3)
			plt.plot(np.arange(len(self.cost_table3)), self.cost_table3)
			plt.ylabel('Cost')
			plt.xlabel('training steps')
			plt.show()

# if __name__ == '__main__':
# 	DQN = DeepQNetwork(3, 4, output_graph=True)



