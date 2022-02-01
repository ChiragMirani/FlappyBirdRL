
#Import Dependencies
import random
import numpy as np
import flappy_bird_gym

#a data structure to
from collections import deque
#layers for deep learning
from keras.layers import Input, Dense

from tensorflow.keras.models import load_model, save_model, Sequential

#optimizer
from tensorflow.keras.optimizers import RMSprop

#Neural Network for Agent.  
# Q-learning uses a look-up table to figure out the best action
# This is not viable when you have a large state table
# A neural network is a function approximation algorithm
# The Q-learning table can be replaced with a neural network
# Neural Network approximates the q-value from a q-table..
# takes input shape and input shape..
def NeuralNetwork(input_shape, output_shape):
    model = Sequential()  # we are going to use the sequential neural network

    #relu is a non-linear activation function
    #512 layers
    model.add(Dense(512, input_shape=input_shape, activation='relu', kernel_initializer='he_uniform'))
    # 256 layers
    model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
    # 64 layers
    model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
    #output shape.., 0 for nothing and 1 for jump
    model.add(Dense(output_shape, activation='linear', kernel_initializer='he_uniform'))

    #RMSProp, popular optimizer in DQN
    model.compile(loss='mse', optimizer=RMSprop(lr=0.0001, rho=0.95, epsilon=0.01), metrics=['accuracy'])
    model.summary()
    return model

#This is the brain of our agent..
class DQNAgent:
    def __init__(self):
        #Environment Variables
        self.env = flappy_bird_gym.make("FlappyBird-v0")
        self.episodes = 1000
        self.state_space = self.env.observation_space.shape[0]
        #jump or not jump
        self.action_space = self.env.action_space.n
        self.memory = deque(maxlen=2000)

        #Hyperparameters 
        # discount rate.. high gamma
        self.gamma = 0.95
        #probability of taking random actions..
        self.epsilon = 1
        # decay epsilon..
        self.epsilon_decay = 0.95

        self.epsilon_min = 0.01

        # amount of data you want to show your agent
        self.batch_number = 64 #16, 32, 128, 256
        
        self.train_start = 1000
        self.jump_prob = 0.01

        #create a neural network
        self.model = NeuralNetwork(input_shape=(self.state_space,), output_shape=self.action_space)

    #how to act appropriately
    def act(self, state):
        # if random variable (between 0 and 1) is bigger than epsilon then 
        # start taking exploitation or learned steps..
        if np.random.random() > self.epsilon:
            #pick the best action by getting the index..
            return np.argmax(self.model.predict(state))
        return 1 if np.random.random() < self.jump_prob else 0

    # learn function: 

    def learn(self):
        #Make sure we have enough data
        if len(self.memory) < self.train_start:
            return

        #Create minibatch
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_number))
        #Variables to store minibatch info
        state = np.zeros((self.batch_number, self.state_space))
        next_state = np.zeros((self.batch_number, self.state_space))
        action, reward, done = [], [], []

        #Store data in variables
        for i in range(self.batch_number):
            state[i] = minibatch[i][0]
            #store action and reward
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        #Predict action based on state
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)
        # Implementing DQN below
        # See https://www.researchgate.net/figure/Pseudo-code-of-DQN-with-experience-replay-method-12_fig11_333197086

        for i in range(self.batch_number):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        #print('train')
        self.model.fit(state, target, batch_size=self.batch_number, verbose=0)

    # train to learn how to play
    def train(self):
        #n episode Iterations for training
        for i in range(self.episodes):
            #Environment variables for training 
            state = self.env.reset()
            #reshape the state variable into a shape that the neural network expects
            state = np.reshape(state, [1, self.state_space])

            done = False
            score = 0
            # researchers have found this equation to be the best in terms of exploiting
            # learned knowledge overtime.  It maintains balance between learning and exploiting.
            self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon * self.epsilon_decay > self.epsilon_min else self.epsilon_min

            while not done:
                #self.env.render()
                # find the action
                action = self.act(state)
                # find out the result of the action
                next_state, reward, done, info = self.env.step(action)

                #reshape nextstate
                next_state = np.reshape(next_state, [1, self.state_space])
                score += 1

                if done:
                    reward -= 100

                self.memory.append((state, action, reward, next_state, done))
                state = next_state
                #keep track of what is done
                if done:
                    print('Episode: {}\nScore: {}\nEpsilon: {:.2}'.format(i, score, self.epsilon))
                    #Save model
                    if score >= 500:
                        self.model.save('flappybrain.h5')
                        return
                # learn function will train our neural network based on our memory
                # this is basically the Q-learning table process..
                self.learn()
                self.model.save('flappybrain.h5')
    #Visualize our model
    def perform(self):
        self.model = load_model('flappybrain.h5')
        while 1:
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_space])
            done = False
            score = 0

            while not done:
                self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, info = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_space])
                score += 1

                print("Current Score: {}".format(score))

                if done: 
                    print('DEAD')
                    break

# main function 
if __name__ == '__main__':
    # create an agent
    agent = DQNAgent()
    #agent.train()
    agent.perform()
