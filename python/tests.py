from cartpole_bernstein import *

def approx_dist(f, g, xs, p=2): # 2-norm default
    if isinstance(xs[0], float) :
        return (sum([abs(f(x) - g(x))**p for x in xs])**(1/p)) / len(xs)
    return (sum([abs(f(*x) - g(*x))**p for x in xs])**(1/p)) / len(xs)

def test_polynomial_add():
    m1 = np.random.rand(2,3)
    m2 = np.random.rand(4,1,5)
    p1 = Polynomial(m1)
    p2 = Polynomial(m2)
    x = np.random.rand(3)
    print('p1(x) + p2(x): {}'.format(p1(x) + p2(x)))
    print('(p1 + p2)(x): {}'.format((p1 + p2)(x)))

def test_monomial_mul():
    m1 = np.random.rand(2,3)
    p1 = Polynomial(m1)
    print('p1 = {}'.format(p1))
    print('p1 * 2 x_0^5 x_3^3 = {}'.format(p1.monomial_mul((5,0,0,3),2)))
    x = np.random.rand(4)
    print('p1(x) * 2 x_0^5 x_3^3 = {}'.format(p1(x) * 2 * x[0]**5 * x[3]**3))
    print('(p1 * (2 x_0^5 x_3^3))(x) = {}'.format(
        p1.monomial_mul((5,0,0,3),2)(x)))

def test_polynomial_mul():
    m1 = np.random.rand(2,4)
    m2 = np.random.rand(3,1,5)
    p1 = Polynomial(m1)
    p2 = Polynomial(m2)
    print('p1 = {}'.format(p1))
    print('p2 = {}'.format(p2))
    print('p1 * p2 = {}'.format(p1 * p2))
    x = np.random.rand(3)
    print('p1(x) * p2(x) = {}'.format(p1(x) * p2(x)))
    print('(p1 * p2)(x) = {}'.format((p1 * p2)(x)))

def plot_bernstein1():
    n = 100 # plotting definition x-axis
    m = 4
    xs = np.linspace(0,1,n)
    ys = np.empty(shape=(n, m + 1))
    fig, ax = plt.subplots()
    for i in range(m + 1): 
        b = bernstein_basis(i, m) * xs[i]
        ys[:,i] = [b(x) for x in xs]
    sum_ys = ys.sum(axis=1)
    ax.plot(xs, ys)
    ax.plot(xs, sum_ys, '--')
    plt.show()

def plot_bernstein2():
    f1 = lambda x: 1-abs(2*x-1) # function to approximate
    f2 = lambda x: (np.pi * x) - np.floor(np.pi * x) # not continuous!
    f3 = lambda x: np.sin(20*x)
    f4 = lambda x: np.piecewise(x, [
        x < 0.3,
        (0.3 <= x) * (x < 0.6),
        0.6 <= x],[
            lambda x_: x_,
            lambda x_: 0.3,
            lambda x_: x_ - 0.3])
    f = f4

    l = 100 # plotting definition x-axis
    n = 12 # approximation definition
    xs = np.linspace(0, 1, l)
    p = bernstein_approx(f, n)

    fig, axs = plt.subplots(2)
    axs[0].plot(xs, [f(x) for x in xs])
    axs[0].plot(xs, [p(x) for x in xs])

    dmax = 20
    ds = np.empty(dmax)
    for n_ in range(dmax):
        p_ = bernstein_approx(f, n_ + 1)
        ds[n_] = approx_dist(f, p_, xs)
    axs[1].plot(ds)

    L = 1
    bs = [L/2 * (1/np.sqrt(n_)) for n_ in range(dmax)]
    axs[1].plot(bs)
    axs[1].set_yscale('log')

    plt.show()

def plot_bernstein3():
    f = lambda x, y: np.sin(3 * x) + np.cos(4 * y)

    dmax = 10
    ds = np.empty(dmax)
    df = 20 #grid definition
    xcs = np.linspace(0,1,df)
    ycs = np.linspace(0,1,df)
    xs = [(x,y) for x in xcs for y in ycs]
    for n in range(dmax):
        p = bernstein_approx(f, (n+1, n+1))
        ds[n] = approx_dist(f, p, xs)
    plt.plot(ds)

    L = 7
    bs = [L/2 * np.sqrt(2/(n+1)) for n in range(dmax)]
    plt.plot(bs)
    plt.yscale('log')
    plt.show()
    #seems to work

plot_bernstein2()


