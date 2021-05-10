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
import threading
import matplotlib.pyplot as plt
# Setting seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(42)
rn.seed(12345)


# BUILDING THE ENVIRONMENT BY SIMPLY CREATING AN OBJECT OF THE ENVIRONMENT CLASS
env = environment.Environment(optimal_temperature = (18.0, 24.0), initial_month = 0, initial_number_users = 20, initial_rate_data = 30)

# LOADING A PRE-TRAINED BRAIN



def Multiple_test(shows):
    # TEST PARAMETER
    # SEEDS
    seeds=[42,43,44]
    seeds2=[12345,12346,12347]
    # RANGE OF USERS
    min_users=[10,100,200]
    max_users=[100,1000,500]
    user_range=[5,50,20]
    # RANGE OF TRANSPORT RATE
    min_rate_data=[20,100,200]
    max_rate_data=[300,600,1000]
    data_range=[10,40,70]
    # TEST TIME RANGE
    years=[12 * 30 * 24 * 60, 24 * 30 * 24 * 60, 36 * 30 * 24 * 60]
    # CHOOSING THE MODE
    train = False
    if __name__ == '__main__':
        for v_seed in range(0,3):
            np.random.seed(seeds[v_seed])
            rn.seed(seeds2[v_seed])
            for v_user in range(0,3):
                for v_rate in range(0,3):
                    # INITIALIZE ENVIRONMENT
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
                    # SET PARAMETERS FOR TEST
                    env.max_rate_data=max_rate_data[v_rate]
                    env.min_rate_data=min_rate_data[v_rate]
                    env.max_update_data=data_range[v_rate]
                    env.max_number_users = max_users[v_user]
                    env.min_number_users = min_users[v_user]
                    env.max_update_users = user_range[v_user]

                    for v_year in range(0,3):
                        file = open('test.txt', 'a')
                        for timestep in range(0, 12 * 30 * 24 * 60):
                            # OBSERVE STATE AND CHOOSE ACTION FOR STATE
                            current_state = tf.expand_dims(tf.convert_to_tensor(current_state), 0)
                            action= tf.squeeze(actor_model(current_state)).numpy()


                            if action< 0:
                                direction = -1
                            else:
                                direction = 1
                            energy_ai = float(abs(action))
                            # IMPLEMENT ACTION IN ENVIRONMENT AND GET NEXT STATE AND REWARD AND GAME_OVER PARAMETER
                            next_state, reward, game_over = env.update_env(direction, energy_ai, int(timestep / (30 * 24 * 60))%12)
                            current_state = next_state
                            # SAVE FIGURES FOR ILLUSTRATION
                            actions.append(action)
                            times.append(time)
                            temperatures.append(env.temperature_ai)

                            ai_energy.append(env.total_energy_ai)
                            energy.append(env.total_energy_noai)

                            ai_overflow.append(env.ai_overflow)
                            overflow.append(env.overflow)

                            time+=0.1
                            i+=1
                            # THIS PART ILLUSTRATES A TEMPERATURE-ACTION FIGURE, YOU CAN SEE THE TEMPERATURE
                            # CHANGES WITH TIME AND THE CORRESPONDING ACTIONS.
                            # WITH VALUE -1, WON'T SHOW THE FIGURE, SET VALUE TO 0 TO SEE THE FIGURE
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
                        # IF THE PARAMETER SHOWS IS TRUE, YOU CAN SEE THE TEMPERATURE-ACTION GIF
                        if shows:
                            # SET MAX POINTS APPEAR IN A FIGURE
                            POINTS = 40
                            # SET TEMPERATURE GRAPH
                            fig, ax = plt.subplots(2, 1)
                            line_temperature, = ax[0].plot(range(POINTS), temperatures[0:40], 'r-x',
                                                           label='temperature of data center')
                            ax[0].set_xlim([0, POINTS])
                            ax[0].set_ylim([-25, 85])
                            ax[0].set_xlabel("time step")
                            ax[0].set_ylabel("Temperature")
                            ax[0].set_autoscale_on(False)
                            ax[0].legend()
                            # SET ACTION GRAPH
                            line_action, = ax[1].plot(range(POINTS), actions[0:40], 'g-^',
                                                      label='The temperature changed by PID algorithm')
                            ax[1].set_xlim([0, POINTS])
                            ax[1].set_ylim([-10, 10])
                            ax[1].set_xlabel("time step")
                            ax[1].set_ylabel("Action")
                            ax[1].set_autoscale_on(False)
                            ax[1].legend()

                            global loop_time
                            loop_time = 0
                            # UPDATE THE Y_DATA OF POINTS IN FIGURE.
                            def show_output(ax):
                                global loop_time
                                loop_time += 1
                                if loop_time + 40 >= len(temperatures):
                                    return

                                line_temperature.set_ydata(temperatures[loop_time:loop_time + 40])
                                line_action.set_ydata(actions[loop_time:loop_time + 40])
                                ax[0].figure.canvas.draw()
                            # SET TIMER TO TRIG THE UPDATE FUNCTION
                            timer = fig.canvas.new_timer(interval=10)
                            timer.add_callback(show_output, ax)
                            timer.start()
                            plt.show()
                        # THIS PART ILLUSTRATES THE TOTAL ENERGY CONSUMPTION AND OVERHEAT RATE
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
                        # SAVING THE TRAINING RESULTS FOR EACH EPOCH

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
# MOST OPERATIONS IN SINGLE TEST ARE SAME AS THAT IN MULTIPLE TEST
temperatures = [None]*50
actions = [None]*50
times = []
ai_energy = []
energy = []
ai_overflow = []
overflow = []
class Single_test_Thread (threading.Thread):
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

    def run(self):
        # INITIALIZE ENVIRONMENT
        train = False
        env.reset(0)
        env.train = train
        current_state, _, _ = env.observe()
        time = 0

        # START SINGLE TEST
        for timestep in range(0, 12 * 30 * 24 * 60):

            current_state = tf.expand_dims(tf.convert_to_tensor(current_state), 0)
            action = tf.squeeze(actor_model(current_state)).numpy()
            if action < 0:
                direction = -1
            else:
                direction = 1
            energy_ai = float(abs(action))

            next_state, reward, game_over = env.update_env(direction, energy_ai, int(timestep / (30 * 24 * 60)) % 12)
            current_state = next_state

            actions.append(action)
            times.append(time)
            temperatures.append(env.temperature_ai)
            time += 1

            ai_energy.append(env.total_energy_ai)
            energy.append(env.total_energy_noai)

            ai_overflow.append(env.ai_overflow)
            overflow.append(env.overflow)

            if timestep % 40 == -1:
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


        print("Total Energy spent with an AI: {:.0f}".format(env.total_energy_ai))
        print("Total Energy spent with no AI: {:.0f}".format(env.total_energy_noai))
        print("ENERGY SAVED: {:.0f} %".format(
            (env.total_energy_noai - env.total_energy_ai) / env.total_energy_noai * 100))




# SAVE THE PATHS OF MODELS
path_list=[]
#GET CURRENT PATH
paths = os.getcwd()
# FIND THE MODELS UNDER THE DOCUMENT
print(">> This is test section, the available models are")
try:
    for file in os.listdir(paths+'\\saved_model'):
        if file.split('_')[0]=='actor':
            path_list.append(file)
            print(file)
except:
    print("wrong path setting")

# IF MODELS EXIST, START THE SHELL
if len(path_list)>0:
    print(">> Two numbers in file name represent id and action boundary respectively")
    print(">> Please input a number to choose one from the list for test")
    while True:
        command = input('>> ').strip().split()
        if len(command)!=1:
            print("Please enter one number")
            continue
        elif command[0].isdigit():
            if int(command[0])>len(path_list)-1:
                print("Index out of range")
            else:
                model_name="./saved_model/" + path_list[int(command[0])]
                boundary = int(model_name.split('_')[-1].split('.')[0])
                DDPG1.upper_bound = boundary;
                DDPG1.lower_bound = -boundary;
                actor_model = DDPG1.get_actor()
                actor_model.load_weights(model_name)

                command=[]
                while len(command) != 1 or not (command[0] == 'y' or command[0] == 'n'):
                    print("Do you want to see the action-temperature figure? y/n")
                    command = input('>> ').strip().split()

                timer=None
                shows=False
                if command[0] == 'y':
                    shows=True
                    POINTS = 40
                    fig, ax = plt.subplots(2, 1)
                    line_temperature, = ax[0].plot(range(POINTS), temperatures[0:40], 'r-x',
                                                   label='temperature of data center')
                    ax[0].set_xlim([0, POINTS])
                    ax[0].set_ylim([-25, 85])
                    ax[0].set_xlabel("time step")
                    ax[0].set_ylabel("Temperature")
                    ax[0].set_autoscale_on(False)
                    ax[0].legend()

                    line_action, = ax[1].plot(range(POINTS), actions[0:40], 'g-^',
                                              label='The temperature changed by DDPG algorithm')
                    ax[1].set_xlim([0, POINTS])
                    ax[1].set_ylim([-boundary-2, boundary+2])
                    ax[1].set_xlabel("time step")
                    ax[1].set_ylabel("Action")
                    ax[1].set_autoscale_on(False)
                    ax[1].legend()

                    global loop_time
                    loop_time = 0


                    def show_output(ax):
                        global loop_time
                        loop_time += 1
                        if loop_time + 40 >= len(temperatures):
                            return

                        line_temperature.set_ydata(temperatures[loop_time:loop_time + 40])
                        line_action.set_ydata(actions[loop_time:loop_time + 40])
                        ax[0].figure.canvas.draw()


                    timer = fig.canvas.new_timer(interval=2)
                    timer.add_callback(show_output, ax)


                command = []
                while len(command)!=1 or not (command[0]=='m' or command[0]=='s'):
                    print("Do you want to start multiple test or single test? m/s")
                    command = input('>> ').strip().split()

                if command[0]=='m':
                    Multiple_test(shows)
                else:
                    thread1 = Single_test_Thread(1, "Thread-1", 1)
                    thread1.start()
                    if shows:
                        timer.start()
                        plt.show()
                    thread1.join()

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


        else:
            print("Please enter one number")
            continue


