import numpy as np
from connectivity_functions import softmax, get_w_pre_post, get_beta
import IPython

epsilon = 1e-5


def log_epsilon(x):

    return np.log(np.maximum(x, epsilon))

class BCPNN:
    def __init__(self, hypercolumns, minicolumns, beta, w, o=None, s=None, a=None, z_pre=None,
                 z_post=None, p_pre=None, p_post=None, p_co=None, G=1.0, tau_m=0.050, g_w=1, g_beta=1,
                 tau_z_pre=0.240, tau_z_post=0.240, tau_p=10.0, tau_a=2.70, g_a=97.0, g_I = 1.0,
                 k=0.0, prng=np.random):
        # Initial values are taken from the paper on memory by Marklund and Lansner.

        # Random number generator
        self.prng = prng

        # Network parameters
        self.hypercolumns = hypercolumns
        self.minicolumns = minicolumns

        self.n_units = self.hypercolumns * self.minicolumns

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

        #  Dynamic Parameters
        self.G = G
        self.tau_m = tau_m
        self.tau_z_pre = tau_z_pre
        self.tau_z_post = tau_z_post
        self.tau_p = tau_p
        self.tau_a = tau_a
        self.k = k
        self.g_a = g_a
        self.g_w = g_w
        self.g_beta = g_beta
        self.g_I = g_I

        # If state variables and parameters are not initialized
        if o is None:
            self.o = np.ones(self.hypercolumns * self.minicolumns) * (1.0 / self.minicolumns)

        if s is None:
            self.s = np.log(np.ones(self.hypercolumns * self.minicolumns) * (1.0 / self.minicolumns))

        if z_pre is None:
            self.z_pre = np.ones_like(self.o) * (1.0 / self.minicolumns)

        if z_post is None:
            self.z_post = np.ones_like(self.o) * (1.0 / self.minicolumns)

        if p_pre is None:
            self.p_pre = np.ones_like(self.o) * (1.0 / self.minicolumns)

        if p_post is None:
            self.p_post = np.ones_like(self.o) * (1.0 / self.minicolumns)

        if p_co is None:
            self.p_co = np.ones((self.o.size, self.o.size)) * (1.0 / self.minicolumns ** 2)

        # Set the adaptation to zeros by default
        self.a = np.zeros_like(self.o)
        # Set the clamping to zero by defalut
        self.I = np.zeros_like(self.o)

    def reset_values(self):
        self.o = np.ones(self.hypercolumns * self.minicolumns) * (1.0 / self.minicolumns)
        self.s = np.log(np.ones(self.hypercolumns * self.minicolumns) * (1.0 / self.minicolumns))
        self.z_pre = np.ones_like(self.o) * (1.0 / self.minicolumns)
        self.z_post = np.ones_like(self.o) * (1.0 / self.minicolumns)
        self.p_pre = np.ones_like(self.o) * (1.0 / self.minicolumns)
        self.p_post = np.ones_like(self.o) * (1.0 / self.minicolumns)
        self.p_co = np.ones((self.beta.size, self.beta.size)) * (1.0 / self.minicolumns ** 2)

        # Set the input to zero
        self.I = np.zeros_like(self.o)

    def randomize_pattern(self):
        self.o = self.prng.rand(self.hypercolumns * self.minicolumns)
        self.s = np.ones_like(self.o) * (1.0 / self.minicolumns)
        self.z_pre = self.prng.rand(self.hypercolumns * self.minicolumns)
        self.z_post = self.prng.rand(self.hypercolumns * self.minicolumns)
        self.p_pre = self.prng.rand(self.hypercolumns * self.minicolumns)
        self.p_post = self.prng.rand(self.hypercolumns * self.minicolumns)
        self.p_co = self.prng.rand(self.hypercolumns * self.minicolumns)

        # Set the input to zero
        self.I = np.zeros_like(self.o)

    def update_discrete(self, N=1):
        for n in range(N):
            self.s = self.beta + np.dot(self.w, self.o)
            self.o = softmax(self.s, t=(1/self.G), minicolumns=self.minicolumns)

    def update_continuous(self, dt=1.0):

        # Updated the probability and the support
        self.s += (dt / self.tau_m) * (self.g_beta * self.beta + self.g_w * np.dot(self.w, self.o)
                                       + self.g_I * log_epsilon(self.I) - self.s - self.g_a * self.a)
        # Softmax
        self.o = softmax(self.s, t=(1/self.G), minicolumns=self.minicolumns)
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

        # If k > 0 update w and beta
        if self.k > epsilon:
            self.w = get_w_pre_post(self.p_co, self.p_pre, self.p_post)
            self.beta = get_beta(self.p_post)

    def run_network_simulation(self, time, I=None, save=False):
        # Load the clamping
        if I is None:
            self.I = np.zeros_like(self.o)
        else:
            self.I = I

        dt = time[1] - time[0]
        # If not saving
        if not save:
            for t in time:
                self.update_continuous(dt)

        # if saving
        if save:
            history_o = np.zeros((time.size, self.beta.size))
            history_s = np.zeros_like(history_o)
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



