
import matplotlib.pyplot as plt
import gym
from polynomial import *

def approx_dist(f, g, xs, p=2): # 2-norm default
    return (sum([abs(f(*x) - g(*x))**p for x in xs])**(1/p)) / len(xs)





def randEpisode():
    try:
        environment = gym.make('CartPole-v0')
        environment.reset()
        for _ in range(100):
            environment.render()
            _, _, done, _ = environment.step(environment.action_space.sample())
            if done : break
    except Exception as e:
        print(e)
    finally: environment.close()


