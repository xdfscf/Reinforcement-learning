# AI for Business - Minimize cost with Deep Q-Learning
# Building the Environment

# Importing the libraries
import numpy as np
import tensorflow as tf
nums1=0
nums2=0
nums3=0
# BUILDING THE ENVIRONMENT IN A CLASS

class Environment(object):
    
    # INTRODUCING AND INITIALIZING ALL THE PARAMETERS AND VARIABLES OF THE ENVIRONMENT
    
    def __init__(self, optimal_temperature = (18.0, 24.0), initial_month = 0, initial_number_users = 10, initial_rate_data = 60):
        # the settled envirment temperatures(maybe it's the only thing that this dqn can learn)
        self.monthly_atmospheric_temperatures = [1.0, 5.0, 7.0, 10.0, 11.0, 20.0, 23.0, 24.0, 22.0, 10.0, 5.0, 1.0]
        # choose initial month
        self.initial_month = initial_month
        # initial temperature
        self.atmospheric_temperature = self.monthly_atmospheric_temperatures[initial_month]
        # the range of optimal temperature
        self.optimal_temperature = optimal_temperature
        # the lower limit of temperature , the server crashes when it exceed the limination of the temperature
        self.min_temperature = -20
        # the uper limit of temperature , the server crashes when it exceed the limination of the temperature
        self.max_temperature = 80
        # the lower limit of users
        self.min_number_users = 10
        # the upper limit of users
        self.max_number_users = 100
        # the upper limit of update users
        self.max_update_users = 5
        # the lower limit of update users
        self.min_rate_data = 20
        # the range of update rate
        self.max_rate_data = 300
        self.max_update_data = 10
        # initialize users
        self.initial_number_users = initial_number_users
        self.current_number_users = initial_number_users
        # initialize rate data
        self.initial_rate_data = initial_rate_data
        self.current_rate_data = initial_rate_data
        # the formula of temperature
        self.intrinsic_temperature = self.atmospheric_temperature + 1.25 * self.current_number_users + 1.25 * self.current_rate_data
        # initialize temperature for ai
        self.temperature_ai = self.intrinsic_temperature
        # temperature_noai=the median value of optimal value boundary
        self.temperature_noai = (self.optimal_temperature[0] + self.optimal_temperature[1]) / 2.0
        # initialize the count of ai energy
        self.total_energy_ai = 0.0
        # initialize the count of noai energy
        self.total_energy_noai = 0.0
        # initialize reward
        self.reward = 0.0
        # The parameter to judge whether the game is the end
        self.game_over = 0
        # The parameter to judge whether train the model
        self.train = 1
        self.proportion=0
        self.overflow=0
    # MAKING A METHOD THAT UPDATES THE ENVIRONMENT RIGHT AFTER THE AI PLAYS AN ACTION
    
    def update_env(self, direction, energy_ai, month):
        
        # GETTING THE REWARD
        
        # Computing the energy spent by the server's cooling system when there is no AI
        energy_noai = 0
        global nums3
        global nums1
        nums1 += 1
        if (self.temperature_noai < self.min_temperature):
                    nums3+=1
                    #self.overflow=nums3/nums1
                    energy_noai = self.optimal_temperature[0] - self.temperature_noai
                    self.temperature_noai = self.optimal_temperature[0]
                    # each time the temperature without being controlled by ai, if less than optimal boundary,change to lower limit of temperature range
        elif (self.temperature_noai > self.max_temperature):
                    nums3+=1
                    #self.overflow=nums3/nums1
                    energy_noai = self.temperature_noai - self.optimal_temperature[1]
                    self.temperature_noai = self.optimal_temperature[1]

        '''
        if (self.temperature_noai < self.optimal_temperature[0]):
            energy_noai = self.optimal_temperature[0] - self.temperature_noai
            self.temperature_noai = self.optimal_temperature[0]
            # each time the temperature without being controlled by ai, if less than optimal boundary,change to lower limit of temperature range
        elif (self.temperature_noai > self.optimal_temperature[1]):
            energy_noai = self.temperature_noai - self.optimal_temperature[1]
            self.temperature_noai = self.optimal_temperature[1]
        '''
        # each time the temperature without being controlled by ai, if higher than optimal boundary,change to upper limit of temperature range
        #print("action: "+str(energy_ai))
        #print("no action: "+str(energy_noai))
        # Computing the Reward
        self.reward = energy_noai - energy_ai
        #print("reward: "+str(self.reward))
        # Scaling the Reward
        self.reward = 1e-3 * self.reward
        
        # GETTING THE NEXT STATE
        
        # Updating the atmospheric temperature
        self.atmospheric_temperature = self.monthly_atmospheric_temperatures[month]
        # Updating the number of users
        self.current_number_users += np.random.randint(-self.max_update_users, self.max_update_users)
        if (self.current_number_users > self.max_number_users):
            self.current_number_users = self.max_number_users
        elif (self.current_number_users < self.min_number_users):
            self.current_number_users = self.min_number_users
        # Updating the rate of data
        self.current_rate_data += np.random.randint(-self.max_update_data, self.max_update_data)
        if (self.current_rate_data > self.max_rate_data):
            self.current_rate_data = self.max_rate_data
        elif (self.current_rate_data < self.min_rate_data):
            self.current_rate_data = self.min_rate_data
        # Computing the Delta of Intrinsic Temperature
        past_intrinsic_temperature = self.intrinsic_temperature
        self.intrinsic_temperature = self.atmospheric_temperature + 1.25 * self.current_number_users + 1.25 * self.current_rate_data
        delta_intrinsic_temperature = self.intrinsic_temperature - past_intrinsic_temperature
        # Computing the Delta of Temperature caused by the AI
        if (direction == -1):
            delta_temperature_ai = -energy_ai
        elif (direction == 1):
            delta_temperature_ai = energy_ai
        # Updating the new Server's Temperature when there is the AI
        self.temperature_ai += delta_intrinsic_temperature + delta_temperature_ai
        #print("ai "+str(self.temperature_ai))
        # Updating the new Server's Temperature when there is no AI
        self.temperature_noai += delta_intrinsic_temperature
        #和当前温度有关，用户量等造成偏置
        #print("noai "+str(self.temperature_noai))
        
        # GETTING GAME OVER

        global nums2

        if (self.temperature_ai < self.min_temperature):
            if (self.train == 1):

                self.game_over = 1
            else:
                self.total_energy_ai += self.optimal_temperature[0] - self.temperature_ai
                self.temperature_ai = self.optimal_temperature[0]
                nums2+=1
                self.overflow=nums2/nums1
                #print("overflow times :"+str(nums2) +" proportion: "+str(nums2/nums1))

        elif (self.temperature_ai > self.max_temperature):

            if (self.train == 1):
                self.game_over = 1

            else:
                self.total_energy_ai += self.temperature_ai - self.optimal_temperature[1]
                self.temperature_ai = self.optimal_temperature[1]
                nums2+=1

                #print("overflow times :" + str(nums2) + " proportion: " + str(nums2 / nums1))
        # UPDATING THE SCORES
        self.overflow = nums2 / nums1

        
        # Updating the Total Energy spent by the AI

        self.total_energy_ai += energy_ai
        # Updating the Total Energy spent by the server's cooling system when there is no AI
        self.total_energy_noai += energy_noai
        
        # SCALING THE NEXT STATE

        scaled_temperature_ai = (self.temperature_ai - self.min_temperature) / (
                    self.max_temperature - self.min_temperature)

        scaled_number_users = (self.current_number_users - self.min_number_users) / (
                    self.max_number_users - self.min_number_users)

        scaled_rate_data = (self.current_rate_data - self.min_rate_data) / (self.max_rate_data - self.min_rate_data)

        state = [self.temperature_ai, scaled_number_users, scaled_rate_data]


        # RETURNING THE NEXT STATE, THE REWARD, AND GAME OVER
        
        return state, self.reward, self.game_over

    # MAKING A METHOD THAT RESETS THE ENVIRONMENT
    
    def reset(self, new_month):
        self.atmospheric_temperature = self.monthly_atmospheric_temperatures[new_month]
        self.initial_month = new_month
        self.current_number_users = self.initial_number_users
        self.current_rate_data = self.initial_rate_data
        self.intrinsic_temperature = self.atmospheric_temperature + 1.25 * self.current_number_users + 1.25 * self.current_rate_data
        self.temperature_ai = self.intrinsic_temperature
        self.temperature_noai = (self.optimal_temperature[0] + self.optimal_temperature[1]) / 2.0
        self.total_energy_ai = 0.0
        self.total_energy_noai = 0.0
        self.reward = 0.0
        self.game_over = 0
        self.overflow=0
        self.train = 1
        global nums1
        global nums2
        global nums3
        nums1=0
        nums2=0
        nums3=0

    # MAKING A METHOD THAT GIVES US AT ANY TIME THE CURRENT STATE, THE LAST REWARD AND WHETHER THE GAME IS OVER
    
    def observe(self):
        scaled_temperature_ai = (self.temperature_ai - self.min_temperature) / (self.max_temperature - self.min_temperature)

        scaled_number_users = (self.current_number_users - self.min_number_users) / (self.max_number_users - self.min_number_users)

        scaled_rate_data = (self.current_rate_data - self.min_rate_data) / (self.max_rate_data - self.min_rate_data)

        state = [self.temperature_ai, scaled_number_users, scaled_rate_data]


        return state, self.reward, self.game_over
