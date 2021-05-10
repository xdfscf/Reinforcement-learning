import numpy as np
import tensorflow as tf

from keras.layers import Input, Dense, Lambda, concatenate
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K

import os
import random
import gym
from collections import deque


import gym
import tensorflow as tf
from tensorflow.python.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import random as rn
import environment

#tf.enable_eager_execution()
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(42)
rn.seed(12345)
#environment instance
env = environment.Environment(optimal_temperature = (18.0, 24.0), initial_month = 0, initial_number_users = 20, initial_rate_data = 30)
#demension of state, this will be used in nerual network
num_states = 3
print("Size of State Space ->  {}".format(num_states))
#demension of action, this will be used in nerual network
num_actions = 1
print("Size of Action Space ->  {}".format(num_actions))
#The upper boundary for action
upper_bound = 10
#The lower boundary for action
lower_bound = -10

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))



# The noises are generated to encourage the exploration of DDPG algorithm
# For each time, the noise will be added to the action chosen by actor network
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)





# The Buffer is used for memory reply, DDPG will learn from these experience sections.
class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.

    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:

            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )
        return actor_loss


    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        state_batch=tf.cast(state_batch, dtype=tf.float32)
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        action_batch=tf.cast(action_batch, dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
        next_state_batch = tf.cast(next_state_batch, dtype=tf.float32)
        loss=self.update(state_batch, action_batch, reward_batch, next_state_batch)
        return loss


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.

def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)
    # The action boundary for actions.
    # Our upper bound is 10.0 for temperature control agent.
    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states,))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions,))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model



# policy will return the action for tranining process
def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action



    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]


"""
## Training hyperparameters
"""
from keras.models import load_model

std_dev = 0.2
#build noise
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))
#build actor and critic
actor_model = get_actor()
critic_model = get_critic()
#load actor and critic
#actor_model.load_weights("try_a_user8.h5")
#critic_model.load_weights("try_c_user8.h5")
#build target actor and critic
target_actor = get_actor()
target_critic = get_critic()
#load target actor or critic
#target_actor.load_weights("try_pa_user8.h5")
#target_critic.load_weights("try_pc_user8.h5")
# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
#target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 8000
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005

buffer = Buffer(50000, 64)



# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

#counter for file saving
counters=0

if __name__ == '__main__':
    losses=[]
    for ep in range(total_episodes):

        new_month = np.random.randint(0, 12)
        env.reset(new_month=new_month)
        game_over = False
        current_state, _, _ = env.observe()
        timestep = 0
        # STARTING THE LOOP OVER ALL THE TIMESTEPS (1 Timestep = 1 Minute) IN ONE EPOCH
        direction=0
        episodic_reward=0
        while ((not game_over) and timestep <= 5 * 30 * 24 * 60):

            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(current_state), 0)

            action = policy(tf_prev_state, ou_noise)
            action[0]=tf.cast(action[0], tf.float32)
            #print(action[0])
            # Recieve state and reward from environment.
            if  action[0]<0:
                direction = -1
            else:
                direction = 1
            energy_ai = float(abs(action[0]))

            # UPDATING THE ENVIRONMENT AND REACHING THE NEXT STATE
            next_state, reward, game_over = env.update_env(direction, energy_ai,
                                                       (new_month + int(timestep / (30 * 24 * 60))) % 12, timestep)


            buffer.record((current_state, action, reward, next_state))
            episodic_reward += reward

            losses.append(buffer.learn())
            update_target(target_actor.variables, actor_model.variables, tau)
            update_target(target_critic.variables, critic_model.variables, tau)
            timestep += 1
            # End this episode when `done` is True
            if game_over:
                break

            current_state = next_state
            #save model for some epochs
            if ep > 600 and ep < 4000 and ep % 60 == 1:
                print("hhh")
                counters += 1
                string1 = "1try_a_user" + str(counters) + ".h5"
                string2 = "1try_c_user" + str(counters) + ".h5"
                string3 = "1try_pa_user" + str(counters) + ".h5"
                string4 = "1try_pc_user" + str(counters) + ".h5"
                actor_model.save_weights(string1)
                critic_model.save_weights(string2)

                target_actor.save_weights(string3)
                target_critic.save_weights(string4)
            elif ep > 4000 and ep % 40 == 1:
                counters += 1
                string1 = "1try_a_user" + str(counters) + ".h5"
                string2 = "1try_c_user" + str(counters) + ".h5"
                string3 = "1try_pa_user" + str(counters) + ".h5"
                string4 = "1try_pc_user" + str(counters) + ".h5"
                actor_model.save_weights(string1)
                critic_model.save_weights(string2)

                target_actor.save_weights(string3)
                target_critic.save_weights(string4)

        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
    #    print("current episode reward: " + episodic_reward)
        avg_reward = np.mean(ep_reward_list[-40:])
        print("\n")
        print("Epoch: {:03d}/{:03d}".format(ep, total_episodes))
        print("Total Energy spent with an AI: {:.2f}".format(env.total_energy_ai))
        print("Total Energy spent with no AI: {:.2f}".format(env.total_energy_noai))
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)

    # Plotting graph
    # Episodes versus Avg. Rewards
    plt.subplot(211)
    plt.plot(losses)
    plt.xlabel("Episode")
    plt.ylabel("loss curve")
    plt.subplot(212)
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()
    '''
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()
    '''
 
    '''
    # Save the weights
    actor_model.save_weights("pendulum_actor5.h5")
    critic_model.save_weights("pendulum_critic5.h5")

    target_actor.save_weights("pendulum_target_actor5.h5")
    target_critic.save_weights("pendulum_target_critic5.h5")
    '''
