import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from gridworld import *

# Util

def iterateN(n, f, start):
    x = start
    for _ in range(n):
        x = f(x)
    return x

def dispValues(valueVector):
    m = np.matrix(np.zeros(shape=(5,5)))
    for i in range(nstates):
        s = states[i]
        m[s] = valueVector[i]
    print(m.transpose()) 

def valueVector2Matrix(valueVector):
    m = np.matrix(np.zeros(shape=(5,5)))
    for i in range(nstates):
        s = states[i]
        m[s] = valueVector[i]
    return m.transpose()

# Some calc.

v0 = np.zeros(nstates)
rPol = lambda s, a: 0.25
v1rPol = tPol(rPol, v0)
#dispValues(v1rPol)
v400rPol = iterateN(400, lambda v: tPol(rPol, v), v0)
dispValues(v400rPol)
v2 = tBell(tBell(v0))
v3 = iterateN(3, tBell, v0)
v4 = iterateN(4, tBell, v0)
v400 = iterateN(400, tBell, v0)
#dispValues(v2)
#dispValues(v400)

# Plotting
m1rPol = valueVector2Matrix(v1rPol)
m400rPol = valueVector2Matrix(v400rPol)
m2 = valueVector2Matrix(v2)
m3 = valueVector2Matrix(v3)
m4 = valueVector2Matrix(v4)
m400 = valueVector2Matrix(v400)

def plotValueGrid(ax, m, bound=25):
    ax.imshow(m,
            cmap = 'coolwarm',
            vmin = -bound,
            vmax = bound,
            #norm = colors.Normalize(vmin = -bound, vmax = bound),
            alpha = 0.4)
    for s in states:
        ax.text(s[0], s[1],
                '{x:.2f}'.format(x = m[s[1], s[0]]),
                horizontalalignment = 'center',
                fontsize='x-small')
        ax.set_xticks([])
    ax.set_yticks([])

vgrids = [m1rPol, m400rPol, m3, m400]
fig, axs = plt.subplots(1,len(vgrids))
for m in range(len(vgrids)):
    plotValueGrid(axs[m], vgrids[m])
axs[0].set_xlabel('$V_{1, \\tau_r}$')
axs[1].set_xlabel('$V_{400, \\tau_r}$')
axs[2].set_xlabel('$V_{3}^*$')
axs[3].set_xlabel('$V_{400}^*$')
plt.show()



