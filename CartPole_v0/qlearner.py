# rewards are calculated as the sum of the negative distance between
# the cart and the origin minus the angle theta minus the
# velocities (ideal is all 0s) - angles are over-weigthed.
# (same for future reward)
# observation is x, x_prime, theta, theta_prime
# action is 0(left) or 1(right)

import gym
import math
import random
import numpy as np
import pandas as pd
import time

def get_feature_range(max_value):
	return np.arange(-max_value, max_value + 2.0*max_value/100.0,
		2.0*max_value/100.0)

def get_feature_bins(feature_max):
	# Discretization in 10 bin narrow in the center and wider on
	# the sides
	# Reduces the state space to 10 values per feature
	distribution = [0,0.3,0.4,0.45,0.48,0.5,0.52,0.55,0.6,0.7,1]
	feature_range = get_feature_range(feature_max)
	return pd.qcut(feature_range, distribution, retbins=True)[1].tolist()

def discretize_feature(feature_bins, value):
	for feature_bin in feature_bins:
		if value < feature_bin: return feature_bin
	# Above upper discrete value
	return feature_bins[-1]

def approximate_observation(x, x_vel, theta, theta_vel):
	return discretize_feature(X_BINS, x), \
		discretize_feature(X_VEL_BINS, x_vel), \
		discretize_feature(THETA_BINS, theta), \
		discretize_feature(THETA_VEL_BINS, theta_vel)

Actions = ['0', '1']

class QLearner(object):
	def __init__(self, epsilon = 1.0, epsilon_decay = 0.01,
			learning_rate = 0.5, gamma = 0.7):
		# State:action => Q value
		self.Q = dict()
		self.rewards = dict()
		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay
		self.epsilon_step = 0
		self.learning_rate = learning_rate
		self.gamma = gamma
	
	def update_epsilon(self):
		if self.epsilon == 0.0: return
		# Still exploring...
		self.epsilon_step += 1
		self.epsilon = math.cos(self.epsilon_step * self.epsilon_decay)
		if self.epsilon < 0.0: self.epsilon = 0.0
	
	def is_explore_step(self):
		if self.epsilon == 0.0:
			return False
		explore = np.random.choice((0, 1),
			p=(1-self.epsilon, self.epsilon))
		return explore == 1
	
	def add_state(self, state):
		self.Q[state] = dict()
		for action_ in Actions:
			self.Q[state][action_] = 0.0
		tup = state
		state = dict(state)
		state_reward = abs(state['x']) - abs(state['x_vel']) \
			- 5 * abs(state['theta']) - 5 * abs(state['theta_vel'])
		self.rewards[tup] = state_reward
	
	def update_Q(self, state, next_state, action):
		old_value = self.Q[state][action]
		self.Q[state][action] = old_value + self.learning_rate * (
			self.rewards[next_state] + self.gamma *
				self.maxQ(next_state) - old_value)
	
	def best_action(self, state):
		max_actions = list()
		max_value = float('-inf')
		for action in self.Q[state].keys():
			value = self.Q[state][action]
			if value > max_value:
				max_actions = [action]
				max_value = value
			elif value == max_value:
				max_actions.append(action)
		if (len(max_actions) == 0):
			print("self.Q[state]: ", self.Q[state])
		assert(len(max_actions) > 0)
		return max_actions[0] if (len(max_actions) == 1) else random.choice(max_actions)
	
	def maxQ(self, state):
		action = self.best_action(state)
		return self.Q[state][action]
	
	def obs_to_state(self, x, x_vel, theta, theta_vel):
		x, x_vel, theta, theta_vel = approximate_observation(x,
				x_vel, theta, theta_vel)
		state = tuple({ 'x': x, 'x_vel': x_vel,
			'theta': theta, 'theta_vel': theta_vel }.items())
		# Adds the state to our cache if not there already
		# so that any state tuple is sure to be cached
		if state not in self.Q.keys():
			self.add_state(state)
		return state
	
	def decide_action(self, state):
		if self.is_explore_step():
			return random.choice(Actions)
		return self.best_action(state)
	
	def dumpQ(self):
		print("Agent.Q size: ", len(agent.Q))
		print("/-----------------------------------------\n")
		print("| State-action rewards from Q-Learning\n")
		print("\-----------------------------------------\n\n")
		for state in self.Q:
			line = ' --'
			for feature, value in dict(state).iteritems():
				line += " {} : {:.2f}".format(feature, value)
			line += '\n'
			for action, reward in agent.Q[state].iteritems():
				line += " -- {} : {:.2f}\n".format(action, reward)
			print(line + '\n')

# Constants Features Ranges
MAX_X			= 2.4
MAX_X_VEL		= 0.1
MAX_THETA		= 12 * 2 * math.pi / 360	# ~ 0.209
MAX_THETA_VEL	= 1.5
X_BINS			= get_feature_bins(MAX_X)
X_VEL_BINS		= get_feature_bins(MAX_X_VEL)
THETA_BINS		= get_feature_bins(MAX_THETA)
THETA_VEL_BINS	= get_feature_bins(MAX_THETA_VEL)

random.seed(47)
env = gym.make('CartPole-v0')
agent = QLearner(epsilon_decay = 0.00001, learning_rate = 1.0)

consecutive_wins = 0
for i_episode in range(20000):
	[x, x_vel, theta, theta_vel] = env.reset()
	state = agent.obs_to_state(x, x_vel, theta, theta_vel)
	for t in range(2000):
		#env.render()
		#time.sleep(0.1)
		action = agent.decide_action(state)
		[x, x_vel, theta, theta_vel], _, done, _ = env.step(int(action))
		if done:
			if t+1 >= 195:
				consecutive_wins += 1
			else:
				consecutive_wins = 0
			print("Episode {:5d} finished after {:4d} timesteps -- epsilon {:.6f}".format(i_episode+1, t+1, agent.epsilon))
			break
		
		next_state = agent.obs_to_state(x, x_vel, theta, theta_vel)
		agent.update_Q(state, next_state, action)
		state = next_state
		agent.update_epsilon()
	if consecutive_wins > 100:
		print("Challenge won after {} episodes".format(i_episode+1))
		break

# Show the states/actions/Qvalues
#agent.dumpQ()
