import gym
import random
import os
import ray
import numpy as np
import multiprocessing as mp
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam

ray.init()

@ray.remote
class FinalNetwork(object):
    def __init__(self):
        self.x = CartPole([], 0)
    
    def update_final_network(self, state, target):
        self.x.agent.brain.fit(state, target, epochs=1, verbose=0)
        
    def save_final_network(self):
        self.x.agent.brain.save(self.x.agent.weights_file)
    

class Agent():
    def __init__(self, state_size, action_size, shared_replay, final_network_actor):
        self.weights_file      = "final_network.h5"
        self.state_size         = state_size
        self.action_size        = action_size
        self.shared_replay      = list(shared_replay)
        self.learning_rate      = 0.001
        self.gamma              = 0.95
        self.exploration_rate   = 1.0
        self.exploration_min    = 0.01
        self.exploration_decay  = 0.995
        self.brain              = self._build_model()
        self.final_network_actor = final_network_actor

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        if os.path.isfile(self.weights_file):
            model.load_weights(self.weights_file)
            self.exploration_rate = self.exploration_min
        return model

    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        act_values = self.brain.predict(state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.shared_replay.append((state, action, reward, next_state, done))

    def replay(self, sample_batch_size):
        if len(self.shared_replay) < sample_batch_size:
            return
        sample_batch = random.sample(self.shared_replay, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
              target = reward + self.gamma * np.amax(self.brain.predict(next_state)[0])
            target_f = self.brain.predict(state)
            target_f[0][action] = target
            self.brain.fit(state, target_f, epochs=1, verbose=0)
            self.final_network_actor.update_final_network.remote(state, target_f)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

class CartPole:
    def __init__(self, l, f):
        self.sample_batch_size = 32
        self.episodes          = 10
        self.env               = gym.make('CartPole-v1')

        self.state_size        = self.env.observation_space.shape[0]
        self.action_size       = self.env.action_space.n
        self.agent             = Agent(self.state_size, self.action_size, l, f)
        self.f_n               = f


    def run(self):
        for index_episode in range(self.episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])

            done = False
            index = 0
            while not done:
                # self.env.render()

                action = self.agent.act(state)

                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                index += 1
            print("Episode {}# Score: {}".format(index_episode, index + 1))
            self.agent.replay(self.sample_batch_size)

def worker(l, f):
    cartpole = CartPole(l, f)
    cartpole.run()

if __name__ == "__main__":
    manager = mp.Manager()
    shared_list = manager.list()
    
    final_network = FinalNetwork.remote()
    
    processes = []
    
    for i in range(3):
        processes.append(mp.Process(target=worker, args=(shared_list, final_network)))
        
    for i in range(3):
        processes[i].start()

    for i in range(3):
        processes[i].join()
        
    print("\n\nTraining is finished.\n\n")
    
    final_network.save_final_network.remote()
    
    print("Neural network saved.\n\n")
    