import numpy as np
from connectivity_functions import softmax
import IPython


class BCPNN:
    def __init__(self, beta, w, o=None, s=None, a=None, z_pre=None, z_post=None,
                 p_pre=None, p_post=None, p_co=None, G=None, tau_m=None):

        # Connectivity
        self.beta = beta
        self.w = w

        # State variables
        self.s = s
        self.o = o
        self.a = a
        self.z_pre = z_pre
        self.z_post = z_post
        self.p_pre = p_pre
        self.p_post = p_post
        self.p_co = p_co

        # Parameters
        self.G = G
        self.tau_m = tau_m

        # If state variables and parameters are not initialized
        if o is None:
            self.o = np.random.rand(beta.size)
        if s is None:
            self.s = np.zeros_like(self.o)

        if G is None:
            self.G = 1.0

        if tau_m is None:
            self.tau_m = 1.0

    def update_discrete(self, N=1):
        for n in range(N):
            self.s = self.beta + np.dot(self.w, self.o)
            self.o = softmax(self.s, t=(1/self.G))

    def update_continuous(self, dt=1.0):
        # IPython.embed()
        self.s += (dt / self.tau_m) * (self.beta + np.dot(self.w, self.o) - self.s)
        self.o = softmax(self.s, t=(1/self.G))


