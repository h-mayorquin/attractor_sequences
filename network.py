import numpy as np
from connectivity_functions import softmax, get_w_pre_post, get_beta, epsilon, log_epsilon
import IPython


class BCPNN:
    def __init__(self, hypercolumns, minicolumns, beta=None, w=None, o=None, s=None, a=None, z_pre=None,
                 z_post=None, p_pre=None, p_post=None, p_co=None, G=1.0, tau_m=0.050, g_w=1, g_beta=1,
                 tau_z_pre=0.240, tau_z_post=0.240, tau_p=10.0, tau_a=2.70, g_a=97.0, g_I=1.0,
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

        if beta is None:
            self.beta = np.log(np.ones_like(self.o) * (1.0 / self.minicolumns))

        if w is None:
            self.w = np.zeros((self.n_units, self.n_units))


        # Set the adaptation to zeros by default
        self.a = np.zeros_like(self.o)
        # Set the clamping to zero by defalut
        self.I = np.zeros_like(self.o)

        # Initialize saving dictionary
        self.history = None
        self.empty_history()

    def empty_history(self):
        """
        A function to empty the history
        """
        empty_array = np.array([]).reshape(0, self.n_units)
        empty_array_square = np.array([]).reshape(0, self.n_units, self.n_units)

        self.history = {'o': empty_array, 's': empty_array, 'z_pre': empty_array,
                        'z_post': empty_array, 'a': empty_array, 'p_pre': empty_array,
                        'p_post': empty_array, 'p_co': empty_array_square, 'w': empty_array_square,
                        'beta': empty_array}

    def reset_values(self, keep_connectivity=False):
        self.o = np.ones(self.n_units) * (1.0 / self.minicolumns)
        self.s = np.log(np.ones(self.n_units) * (1.0 / self.minicolumns))
        self.z_pre = np.ones_like(self.o) * (1.0 / self.minicolumns)
        self.z_post = np.ones_like(self.o) * (1.0 / self.minicolumns)
        self.p_pre = np.ones_like(self.o) * (1.0 / self.minicolumns)
        self.p_post = np.ones_like(self.o) * (1.0 / self.minicolumns)
        self.p_co = np.ones((self.beta.size, self.beta.size)) * (1.0 / self.minicolumns ** 2)

        self.a = np.zeros_like(self.o)

        if not keep_connectivity:
            self.beta = np.log(np.ones_like(self.o) * (1.0 / self.minicolumns))
            self.w = np.zeros((self.n_units, self.n_units))

    def randomize_pattern(self):
        self.o = self.prng.rand(self.n_units)
        self.s = np.log(self.prng.rand(self.n_units))

        # A follow o, if o is randomized sent a to zero.
        self.a = np.zeros_like(self.o)

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
        self.p_co += (dt / self.tau_p) * (np.outer(self.z_pre, self.z_post) - self.p_co) * self.k
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
            history_o = np.zeros((time.size, self.n_units))
            history_s = np.zeros_like(history_o)
            history_z_pre = np.zeros_like(history_o)
            history_z_post = np.zeros_like(history_o)
            history_a = np.zeros_like(history_o)
            history_p_pre = np.zeros_like(history_o)
            history_p_post = np.zeros_like(history_o)
            history_p_co = np.zeros((time.size, self.n_units, self.n_units))
            history_beta = np.zeros_like(history_o)
            history_w = np.zeros_like(history_p_co)

            for index_t, t in enumerate(time):
                history_o[index_t, :] = self.o
                history_s[index_t, :] = self.s
                history_z_pre[index_t, :] = self.z_pre
                history_z_post[index_t, :] = self.z_post
                history_a[index_t, :] = self.a
                history_p_pre[index_t, :] = self.p_pre
                history_p_post[index_t, :] = self.p_post
                history_p_co[index_t, ...] = self.p_co

                history_beta[index_t, :] = self.beta
                history_w[index_t, ...] = self.w

                # Update the system
                self.update_continuous(dt)

            # Concatenate with the past and redefine dictionary
            self.history['o'] = np.concatenate((self.history['o'], history_o))
            self.history['s'] = np.concatenate((self.history['s'], history_s))
            self.history['z_pre'] = np.concatenate((self.history['z_pre'], history_z_pre))
            self.history['z_post'] = np.concatenate((self.history['z_post'], history_z_post))
            self.history['a'] = np.concatenate((self.history['a'], history_a))
            self.history['p_pre'] = np.concatenate((self.history['p_pre'], history_p_pre))
            self.history['p_post'] = np.concatenate((self.history['p_post'], history_p_post))
            self.history['p_co'] = np.concatenate((self.history['p_co'], history_p_co))

            self.history['w'] = np.concatenate((self.history['w'], history_w))
            self.history['beta'] = np.concatenate((self.history['beta'], history_beta))


            return self.history



