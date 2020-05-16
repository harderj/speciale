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

def fourGrids():
    vgrids = [m1rPol, m400rPol, m3, m400]
    fig, axs = plt.subplots(1,len(vgrids))
    for m in range(len(vgrids)):
        plotValueGrid(axs[m], vgrids[m])
    axs[0].set_xlabel('$V_{1, \\tau_r}$')
    axs[1].set_xlabel('$V_{400, \\tau_r}$')
    axs[2].set_xlabel('$V_{3}^*$')
    axs[3].set_xlabel('$V_{400}^*$')
    plt.show()

def convGraph():
    n = 100
    xs = np.array(range(n))
    voptmax = max(v400) # |V*|
    vtauroptmax = max(v400rPol) # |V_taur|
    gys = gamma**xs * 100 # general theoretical upper bound
    vys = gamma**xs * voptmax # theoretical upper bound for V^*
    tys = gamma**xs * vtauroptmax # theoretical upper bound for V_taur
    ays = np.zeros(n) # |V*k - V*|
    rys = np.zeros(n) # |V_{taur,k} - V_taur|
    v = np.zeros(nstates)
    vr = np.zeros(nstates)
    for i in range(n):
        v = tBell(v)
        vr = tPol(rPol, vr)
        ays[i] = max(abs(v - v400))
        rys[i] = max(abs(vr - v400rPol))
    fig, ax = plt.subplots()
    blue1 = '#1f77b4' # 'category10 blue' (standard f. pyplot, Tableau, a.o.)
    orange1 = '#ff7f0e' # 'category10 orange'
    ax.plot(xs, ays, color = blue1)
    ax.plot(xs, rys, color = orange1)
    ax.plot(xs, vys, ls=':', color = blue1)
    ax.plot(xs, tys, ls=':', color = orange1)
    ax.plot(xs, gys, ls='--', color='black')
    ax.set_yscale('log')
    ax.legend(['$|V_k^* - V_{400}^*|$',
        '$|V_{\\tau_r, k} - V_{\\tau_r, 400}|$',
        '$\gamma^k |V_{400}^*|$',
        '$\gamma^k |V_{\\tau_r, 400}|$',
        '$\gamma^k V_{\max} / (1-\gamma)$'])
    ax.set_xlabel('$k$')
    ax.set_ylabel('value (expected acc. rewards)')
    plt.show()

convGraph()



