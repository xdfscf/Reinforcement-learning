# AI for Business - Minimize cost with Deep Q-Learning
# Testing the AI

# Installing Keras
# conda install -c conda-forge keras

# Importing the libraries and the other python files
import os
import numpy as np
import random as rn
from keras.models import load_model
import environment
import DDPG1
import tensorflow as tf

import matplotlib.pyplot as plt
# Setting seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(42)
rn.seed(12345)

# SETTING THE PARAMETERS
number_actions = 5
direction_boundary = (number_actions - 1) / 2
temperature_step = 1.5

# BUILDING THE ENVIRONMENT BY SIMPLY CREATING AN OBJECT OF THE ENVIRONMENT CLASS
env = environment.Environment(optimal_temperature = (18.0, 24.0), initial_month = 0, initial_number_users = 20, initial_rate_data = 30)

# LOADING A PRE-TRAINED BRAIN
#model = load_model("new_new_model_harder.h5")
actor_model = DDPG1.get_actor()
actor_model.load_weights("pendulum_actor3.h5")
# CHOOSING THE MODE
train = False
seeds=[42,43,44]
seeds2=[12345,12346,12347]

min_users=[10,100,200]
max_users=[100,1000,500]
user_range=[5,50,20]

min_rate_data=[20,100,200]
max_rate_data=[300,600,1000]
data_range=[10,40,70]

years=[12 * 30 * 24 * 60, 24 * 30 * 24 * 60, 36 * 30 * 24 * 60]
# RUNNING A 1 YEAR SIMULATION IN INFERENCE MODE
'''
if __name__ == '__main__':
    for v_seed in range(0,3):
        np.random.seed(seeds[v_seed])
        rn.seed(seeds2[v_seed])
        for v_user in range(0,3):
            for v_rate in range(0,3):
                env.reset(0)
                env.train = train
                current_state, _, _ = env.observe()

                ai_energy = []
                energy = []

                ai_overflow = []
                overflow = []

                actions = []
                times = []
                temperatures = []
                time = 0
                i = 0

                env.max_rate_data=max_rate_data[v_rate]
                env.min_rate_data=min_rate_data[v_rate]
                env.max_update_data=data_range[v_rate]
                env.max_number_users = max_users[v_user]
                env.min_number_users = min_users[v_user]
                env.max_update_users = user_range[v_user]

                for v_year in range(0,3):
                    file = open('test.txt', 'a')
                    for timestep in range(0, 12 * 30 * 24 * 60):
                        
                        #q_values = model.predict(current_state)
                        #action = np.argmax(q_values[0])
                        
                        current_state = tf.expand_dims(tf.convert_to_tensor(current_state), 0)
                        action= tf.squeeze(actor_model(current_state)).numpy()

                        #print(action)
                        if action< 0:
                            direction = -1
                        else:
                            direction = 1
                        energy_ai = float(abs(action))
                        #print(energy_ai*direction)
                        next_state, reward, game_over = env.update_env(direction, energy_ai, int(timestep / (30 * 24 * 60))%12)
                        current_state = next_state

                        #actions.append((action-direction_boundary)*1.5)
                        actions.append(action)
                        times.append(time)
                        temperatures.append(env.temperature_ai)

                        ai_energy.append(env.total_energy_ai)
                        energy.append(env.total_energy_noai)

                        ai_overflow.append(env.ai_overflow)
                        overflow.append(env.overflow)

                        time+=0.1
                        i+=1

                        if timestep%40==-1:
                            plt.subplot(211)
                            plt.plot(times,temperatures)
                            plt.title('Time_reward-seeds_'+str(seeds[v_seed])+'-'+str(seeds2[v_seed])+'_user-range_'+str(env.min_number_users)+'-'+str(env.max_number_users)+'_user-update_'+str(env.max_update_users)+'_data-range_'+str(env.min_rate_data)+'-'+str(env.max_rate_data)+'_data-update_'+str(env.max_update_data)+'_year'+str(v_year+1), fontsize=12, color='r')
                            plt.xlabel("time step")
                            plt.ylabel("Temperature")
                            plt.subplot(212)
                            plt.plot(times,actions)
                            plt.xlabel("time step")
                            plt.ylabel("Action")
                            plt.show()

                    plt.subplot(211)
                    plt.plot(times, ai_energy, color='r', label='ai controller', linewidth=2)
                    plt.plot(times, energy, color='g', label='traditional controller', linewidth=2)
                    plt.xlabel("time step")
                    plt.ylabel("energy consumption")

                    plt.legend()
                    plt.subplot(212)
                    plt.plot(times, ai_overflow, color='r', label='ai controller', linewidth=2)
                    plt.plot(times, overflow, color='g', label='traditional controller', linewidth=2)
                    plt.xlabel("time step")
                    plt.ylabel("Overflow times")
                    plt.legend()
                    plt.savefig(fname='Time_reward-seeds_'+str(seeds[v_seed])+'-'+str(seeds2[v_seed])+'_user-range_'+str(env.min_number_users)+'-'+str(env.max_number_users)+'_user-update_'+str(env.max_update_users)+'_data-range_'+str(env.min_rate_data)+'-'+str(env.max_rate_data)+'_data-update_'+str(env.max_update_data)+'_year'+str(v_year+1),figsize=[10,10])
                    plt.clf()
                    # PRINTING THE TRAINING RESULTS FOR EACH EPOCH

                    file.write('Time_reward-seeds_'+str(seeds[v_seed])+'-'+str(seeds2[v_seed])+'_user-range_'+str(env.min_number_users)+'-'+str(env.max_number_users)+'_user-update_'+str(env.max_update_users)+'_data-range_'+str(env.min_rate_data)+'-'+str(env.max_rate_data)+'_data-update_'+str(env.max_update_data)+'_year'+str(v_year+1))
                    file.write("\n")
                    file.write("Total Energy spent with an AI: {:.0f}".format(env.total_energy_ai))
                    file.write("\n")
                    file.write("Total Energy spent with no AI: {:.0f}".format(env.total_energy_noai))
                    file.write("\n")
                    file.write("ENERGY SAVED: {:.0f} %".format((env.total_energy_noai - env.total_energy_ai) / env.total_energy_noai * 100))
                    file.write("\n")
                    file.write("overflow times :" + str(env.nums2) + " proportion: " + str(env.ai_overflow))
                    file.write("\n")
                    file.close()
                    print("\n")
                    print("Total Energy spent with an AI: {:.0f}".format(env.total_energy_ai))
                    print("Total Energy spent with no AI: {:.0f}".format(env.total_energy_noai))
                    print("ENERGY SAVED: {:.0f} %".format((env.total_energy_noai - env.total_energy_ai) / env.total_energy_noai * 100))
'''
env.reset(0)
env.train = train
current_state, _, _ = env.observe()
actions = []
times = []
temperatures = []
time = 0
ai_energy = []
energy = []

ai_overflow = []
overflow = []
for timestep in range(0, 12 * 30 * 24 * 60):

    current_state = tf.expand_dims(tf.convert_to_tensor(current_state), 0)
    action = tf.squeeze(actor_model(current_state)).numpy()

    # print(action)
    if action < 0:
        direction = -1
    else:
        direction = 1
    energy_ai = float(abs(action))
    # print(energy_ai*direction)
    next_state, reward, game_over = env.update_env(direction, energy_ai, int(timestep / (30 * 24 * 60)) % 12,timestep)
    current_state = next_state

    actions.append(action)
    times.append(time)
    temperatures.append(env.temperature_ai)
    time+=1

    ai_energy.append(env.total_energy_ai)
    energy.append(env.total_energy_noai)

    ai_overflow.append(env.ai_overflow)
    overflow.append(env.overflow)

    if timestep % 40 == 0:
        plt.subplot(211)
        plt.title("DDPG with action boundary=10")
        plt.plot(times, temperatures, 'r-x', label='temperature of server')
        plt.xlabel("time step")
        plt.ylabel("Temperature")
        plt.legend()
        plt.subplot(212)
        plt.plot(times, actions, 'g-^', label='The temperature changed by agent')
        plt.xlabel("time step")
        plt.ylabel("Action")
        plt.legend()
        plt.show()

plt.subplot(211)
plt.plot(times, ai_energy, color='r', label='ai controller', linewidth=2)
plt.plot(times, energy, color='g', label='traditional controller', linewidth=2)
plt.xlabel("time step")
plt.ylabel("energy consumption")

plt.legend()
plt.subplot(212)
plt.plot(times, ai_overflow, color='r', label='ai controller', linewidth=2)
plt.plot(times, overflow, color='g', label='traditional controller', linewidth=2)
plt.xlabel("time step")
plt.ylabel("Overflow times")
plt.legend()
plt.show()
print("Total Energy spent with an AI: {:.0f}".format(env.total_energy_ai))
print("Total Energy spent with no AI: {:.0f}".format(env.total_energy_noai))
print("ENERGY SAVED: {:.0f} %".format((env.total_energy_noai - env.total_energy_ai) / env.total_energy_noai * 100))




