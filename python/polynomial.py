import numpy as np

class Polynomial:
    def __init__(self, coeffs):
        if isinstance(coeffs, (float, int)):
            self.coeffs = np.array([coeffs])
            return
        if isinstance(coeffs, list):
            self.coeffs = np.array(coeffs)
            return
        self.coeffs = coeffs.copy()

    def __copy__(self):
        return Polynomial(self.coeffs.copy())

    copy = __copy__

    def __call__(self, xs, *xs_):
        sum_ = 0
        #print(xs_)
        if isinstance(xs, (float, int)):
            xs = [xs]
        if len(xs_) > 0:
            xs = np.array([x for x in xs] + [x for x in xs_])
        if not isinstance(xs, (list, tuple, np.ndarray)):
            raise Exception("Input not list, tuple or numpy.ndarray.")
        if len(xs) < len(self.coeffs.shape) :
            raise Exception("Input dimension too low.")
        for i, c in np.ndenumerate(self.coeffs):
            term = c
            for j in range(len(self.coeffs.shape)):
                term *= xs[j]**i[j]
            sum_ += term
        return sum_

    def __str__(self):
        str_ = ""
        for i, c in np.ndenumerate(self.coeffs):
            if c != 0:
                str_ += "{c:.3g} ".format(c=c)
                for j in range(len(self.coeffs.shape)):
                    if i[j] != 0:
                        if i[j] == 1:
                            str_ += "x_{n} ".format(n=j)
                        else:
                            str_ += "x_{n}^{p} ".format(p=i[j], n=j)
                str_ += "+ "
        str_ = str_.replace('+ -', '- ')
        str_ = str_.rstrip('+ ')
        if str_ == "": str_ = "0"
        return str_

    __repr__ = __str__

    def __add__(p, q):
        if isinstance(q, (float, int)):
            return p + Polynomial(np.array([q]))
        pd = len(p.coeffs.shape)
        qd = len(q.coeffs.shape)
        dmax = max(pd, qd)
        pshape = tuple(list(p.coeffs.shape) + [1 for _ in range(dmax - pd)])
        qshape = tuple(list(q.coeffs.shape) + [1 for _ in range(dmax - qd)])
        maxshape = tuple(max(x,y) for (x,y) in zip(pshape, qshape))
        res = np.zeros(shape=maxshape)
        for i, _ in np.ndenumerate(res):
            if all(np.array(i) < np.array(pshape)):
                res[i] += p.coeffs[i[:len(p.coeffs.shape)]]
            if all(np.array(i) < np.array(qshape)):
                res[i] += q.coeffs[i[:len(q.coeffs.shape)]]
        return Polynomial(res)

    __radd__ = __add__

    def __sub__(p, q): return p + ((-1) * q)

    def __rsub__(p, q): return (-1) * (p - q)

    def monomial_mul(p, i, c):
        dmax = max(len(p.coeffs.shape), len(i))
        pshape = tuple(list(p.coeffs.shape)
                + [1 for _ in range(dmax - len(p.coeffs.shape))])
        ishape = tuple(list(i) + [0 for _ in range(dmax - len(i))])
        rshape = tuple(x + y for (x,y) in zip(pshape, ishape))
        res = np.zeros(shape=rshape)
        for j, _ in np.ndenumerate(res):
            jv = np.array(j)
            iv = np.array(ishape)
            if all(jv >= iv):
                res[j] = c * p.coeffs[tuple(jv-iv)[:len(p.coeffs.shape)]]
        return Polynomial(res)

    def __mul__(p, q):
        if isinstance(q, (float, int)):
            return p * Polynomial(np.array([q]))
        dmax = max(len(p.coeffs.shape), len(q.coeffs.shape))
        pshape = tuple(list(p.coeffs.shape)
                + [1 for _ in range(dmax - len(p.coeffs.shape))])
        qshape = tuple(list(q.coeffs.shape)
                + [1 for _ in range(dmax - len(q.coeffs.shape))])
        rshape = tuple(x + y - 1 for (x,y) in zip(pshape, qshape))
        r = Polynomial(np.zeros(shape=rshape))
        for i, _ in np.ndenumerate(q.coeffs):
            r += p.monomial_mul(i, q.coeffs[i])
        return r
    __rmul__ = __mul__

    def __pow__(self, n):
        if n == 0:
            return Polynomial(np.array([1]))
        if n == 1:
            return self.copy()
        else: return self * self.__pow__(n-1)

def x_(n):
    shape = tuple(1 + (i==n) for i in range(n+1))
    index = tuple(int(i==n) for i in range(n+1))
    xs = np.zeros(shape=shape)
    xs[index] = 1
    return Polynomial(xs)

def binom(n, k):
    f = np.math.factorial
    return f(n)/(f(k) * f(n-k))

def bernstein_basis(ks, ns):
    if isinstance(ks, int): ks = [ks]
    if isinstance(ns, int): ns = [ns]
    m = min(len(ns), len(ks))
    p = Polynomial(np.array([1]))
    for j in range(m):
        p *= binom(ns[j], ks[j]) * x_(j)**ks[j]
        p *= (1 - x_(j)) ** (ns[j] - ks[j])
    return p

def bernstein_approx(f, ns):
    if isinstance(ns, int):
        ns = [ns]
    m = len(ns)
    xs = np.empty(shape= tuple(ns[i] + 1 for i in range(m)))
    p = Polynomial(np.array([0]))
    for i, _ in np.ndenumerate(xs):
        c = f( *tuple(i[k]/ns[k] for k in range(m)) )
        p += c * bernstein_basis(i, ns)
    return p

