import numpy as np
from connectivity_functions import softmax, get_w_pre_post, get_beta
import IPython

epsilon = 1e-10


def log_epsilon(x):

    return np.log(np.maximum(x, epsilon))

class BCPNN:
    def __init__(self, beta, w, o=None, s=None, a=None, z_pre=None, z_post=None,
                 p_pre=None, p_post=None, p_co=None, G=None, tau_m=None, tau_z_pre=None,
                 tau_z_post=None, tau_p=None, tau_a=None, g_a=None, k=0, M=1.0):

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
        self.M = M

        # Parameters
        self.G = G
        self.tau_m = tau_m
        self.tau_z_pre = tau_z_pre
        self.tau_z_post = tau_z_post
        self.tau_p = tau_p
        self.tau_a = tau_a
        self.k = k
        self.g_a = g_a

        # If state variables and parameters are not initialized
        if o is None:
            self.o = np.random.rand(beta.size)
        if s is None:
            self.s = np.zeros_like(self.o)

        if z_pre is None:
            self.z_pre = np.ones_like(o) * (1.0 / self.M)

        if z_post is None:
            self.z_post = np.ones_like(o) * (1.0 / self.M)

        if p_pre is None:
            self.p_pre = np.ones_like(o) * (1.0 / self.M)

        if p_post is None:
            self.p_post = np.ones_like(o) * (1.0 / self.M)

        if p_co is None:
            self.p_co = np.ones((beta.size, beta.size)) * (1.0 / self.M**2)

        if G is None:
            self.G = 1.0

        if g_a is None:
            self.g_a = 97.0

        if tau_m is None:
            self.tau_m = 0.050

        if tau_z_pre is None:
            self.tau_z_pre = 0.240

        if tau_z_post is None:
            self.tau_z_post = 0.240

        if tau_p is None:
            self.tau_p = 1.0

        if tau_a is None:
            self.tau_a = 1.0

        self.a = np.zeros_like(self.o)

    def update_discrete(self, N=1):
        for n in range(N):
            self.s = self.beta + np.dot(self.w, self.o)
            self.o = softmax(self.s, t=(1/self.G))

    def update_continuous(self, dt=1.0):
        # Updated the probability and the support
        self.s += (dt / self.tau_m) * (self.beta + np.dot(self.w, self.o) - self.s - self.g_a * self.a)
        self.o = softmax(self.s, t=(1/self.G))
        # Update the adaptation
        self.a += (dt / self.tau_a) * (self.o - self.a)
        # Updated the z-traces
        self.z_pre += (dt / self.tau_z_pre) * (self.o - self.z_pre)
        self.z_post += (dt / self.tau_z_post) * (self.o - self.z_post)
        # Updated the probability
        self.p_pre += (dt / self.tau_p) * (self.z_pre - self.p_pre) * self.k
        self.p_post += (dt / self.tau_p) * (self.z_post - self.p_post) * self.k
        self.p_co += (dt / self.tau_p) * (self.z_pre * self.z_post - self.p_co) * self.k
        # Update probability
        # IPython.embed()

        self.w = get_w_pre_post(self.p_co, self.p_pre, self.p_post)
        self.beta = get_beta(self.p_post)

    def run_network_simulation(self, time, save=False):

        dt = time[1] - time[0]

        # If not saving
        if not save:
            for t in time:
                self.update_continuous(dt)

        # if saving
        if save:
            history_o =  np.zeros((time.size, self.beta.size))
            history_s =  np.zeros_like(history_o)
            history_z_pre = np.zeros_like(history_o)
            history_z_post = np.zeros_like(history_o)
            history_a = np.zeros_like(history_o)

            for index_t, t in enumerate(time):
                history_o[index_t, :] = self.o
                history_s[index_t, :] = self.s
                history_z_pre[index_t, :] = self.z_pre
                history_z_post[index_t, :] = self.z_post
                history_a[index_t, :] = self.a
                # Update the system
                self.update_continuous(dt)

            history_dic = {'o': history_o, 's': history_s, 'z_pre': history_z_pre,
                           'z_post':history_z_post, 'a': history_a}

            return history_dic



