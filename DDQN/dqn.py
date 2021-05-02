# AI for Business - Minimize cost with Deep Q-Learning
# Implementing Deep Q-Learning with Experience Replay

# Importing the libraries
import numpy as np
timestep=0
# IMPLEMENTING DEEP Q-LEARNING WITH EXPERIENCE REPLAY

class DQN(object):
    
    # INTRODUCING AND INITIALIZING ALL THE PARAMETERS AND VARIABLES OF THE DQN
    def __init__(self, max_memory = 400, discount = 0.9):
        #create a list to save periods(period is a list that contain states
        self.memory = list()
        #the max size of saved periods
        self.max_memory = max_memory
        # the value of discount factor
        self.discount = discount


    # MAKING A METHOD THAT BUILDS THE MEMORY IN EXPERIENCE REPLAY
    def remember(self, transition, game_over):
        #save periods into memory
        self.memory.append([transition, game_over])
        #if the number of periods is out of the size od memory,we save it into memory 0
        #why not use a LRU list
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    # MAKING A METHOD THAT BUILDS TWO BATCHES OF INPUTS AND TARGETS BY EXTRACTING TRANSITIONS FROM THE MEMORY
    def get_batch(self, model,model1, batch_size = 10):
        global timestep
        timestep+=1
        if timestep%1000==0:
            model1.set_weights(model.get_weights())

        len_memory = len(self.memory)
        #the size of a state
        num_inputs = self.memory[0][0][0].shape[1]
        #print(self.memory)
        #print(num_inputs)
        #the size of output estimates
        num_outputs = model.output_shape[-1]
        #build a matrix for a batch of input states
        inputs = np.zeros((min(len_memory, batch_size), num_inputs))
        #build a matrix for a batch of output estimates
        targets = np.zeros((min(len_memory, batch_size), num_outputs))
        #ergotic the whole list,using a enumerate method,which is different from list
            #get the parameters of a period
            current_state, action, reward, next_state = self.memory[idx][0]
            #and also the parameter that judge whether a period is terminaled
            game_over = self.memory[idx][1]


            #put state into matrix
            inputs[i] = current_state
            #use model to estimate the current_state of batch periods
            targets[i] = model.predict(current_state)[0]
            #print(model.predict(current_state)[0])
            #use model to estimate the next_state of batch periods
            Q_sa = model1.predict(next_state)[0]
            next_action = np.argmax(model.predict(next_state)[0])
            #the Q(current_state,action)=reward+discount*Max(Q(next_state,action))
            #and the loose equals DQN(current_state,action)-reward-discount*Max(DQN(next_state,action))
            if game_over:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.discount * Q_sa[next_action]

        return inputs, targets
