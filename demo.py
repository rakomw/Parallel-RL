import gym
import numpy as np
from keras import models
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

#change to the desired environment and corresponding weights file
atari = make_atari("SpaceInvadersNoFrameskip-v4")
env = wrap_deepmind(atari, episode_life=False, clip_rewards=False, frame_stack=True, scale=True)
model = models.load_model("spaceweights.h5")
model.compile()

num_episodes = 5

for i in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        env.render()
        state = np.expand_dims(state, 0)
        expected = model.predict(state)
        action = np.argmax(expected[0])

        next_state, reward, done, _ = env.step(action)
        state = next_state
    env.close()
