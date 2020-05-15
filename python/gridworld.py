# Gridworld
# J.Harder 2020-05-15
# value iteration over David Silver's GridWorld example
# see e.g. https://medium.com/@digankate26/coding-the-gridworld-example-from-deepminds-reinforcement-learning-course-in-python-17d74335fcbc

import numpy as np
from typing import Tuple

# Util

def clamp(a,b,x):
    return min(max(x, a), b)

# Gridworld

State = Tuple[int, int]
Action = int # 0 - up, 1 - down, 2 - left, 3 - right

states = [(x,y) for y in range(5) for x in range(5)]
actions = [a for a in range(4)]
gamma = 0.9

nstates = len(states)
nactions = len(actions)

def transFun(state : State, action : Action):
    if state == (1, 0) : return (1, 4)
    if state == (3, 0) : return (3, 2)
    return {
            0: (state[0], clamp(0, 4, state[1] - 1)),
            1: (state[0], clamp(0, 4, state[1] + 1)),
            2: (clamp(0, 4, state[0] - 1), state[1]),
            3: (clamp(0, 4, state[0] + 1), state[1]),
    }[action]

def expRew(state : State, action : Action):
    if state == (1, 0) : return 10
    if state == (3, 0) : return 5
    if transFun(state, action) == state : return -1
    return 0

def tPol(policy, valueVector):
    v1 = valueVector
    v2 = np.zeros(nstates)
    for i in range(nstates):
        s1 = states[i]
        for j in range(nactions):
            a1 = actions[j]
            s2 = transFun(s1, a1)
            k = states.index(s2)
            dt = policy(s1, a1)
            r = expRew(s1, a1)
            v2[i] += (r + gamma * v1[k]) * dt
    return v2

def tBell(valueVector):
    v1 = valueVector
    v2 = np.zeros(nstates)
    for i in range(nstates):
        s1 = states[i]
        x = -np.inf
        for j in range(nactions):
            a1 = actions[j]
            s2 = transFun(s1, a1)
            k = states.index(s2)
            r = expRew(s1, a1)
            x = max(x, r + gamma * v1[k])
        v2[i] = x 
    return v2



