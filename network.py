import numpy as np
from connectivity_functions import softmax, get_w_pre_post, get_beta, epsilon, log_epsilon
import IPython


class BCPNN:
    def __init__(self, hypercolumns, minicolumns, beta=None, w=None, o=None, s=None, a=None, z_pre=None,
                 z_post=None, p_pre=None, p_post=None, p_co=None, G=1.0, tau_m=0.050, g_w=1, g_beta=1,
                 tau_z_pre=0.240, tau_z_post=0.240, tau_p=10.0, tau_a=2.70, g_a=97.0, g_I=10.0,
                 k=0.0, sigma=1.0, prng=np.random):
        # Initial values are taken from the paper on memory by Marklund and Lansner.

        # Random number generator
        self.prng = prng
        self.sigma = sigma

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
            self.p_pre = np.zeros_like(self.o)

        if p_post is None:
            self.p_post = np.zeros_like(self.o)

        if p_co is None:
            self.p_co = np.zeros((self.o.size, self.o.size))

        if beta is None:
            self.beta = np.log(np.ones_like(self.o) * (1.0 / self.minicolumns))

        if w is None:
            self.w = np.zeros((self.n_units, self.n_units))

        # Set the coactivations to a default
        self.z_co = np.ones((self.n_units, self.n_units)) * (1.0 / self.minicolumns ** 2)
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

    def get_parameters(self):
        """
        Get the parameters of the model

        :return: a dictionary with the parameters
        """
        parameters = {'tau_m': self.tau_m, 'tau_z_post': self.tau_z_post, 'tau_z_pre': self.tau_z_post,
                      'tau_p': self.tau_p, 'tau_a': self.tau_a, 'g_a': self.g_a, 'g_w': self.g_w,
                      'g_beta': self.g_beta, 'g_I':self.g_I, 'sigma':self.sigma, 'k': self.k}

        return parameters

    def reset_values(self, keep_connectivity=False):
        self.o = np.ones(self.n_units) * (1.0 / self.minicolumns)
        self.s = np.log(np.ones(self.n_units) * (1.0 / self.minicolumns))
        self.z_pre = np.ones_like(self.o) * (1.0 / self.minicolumns)
        self.z_post = np.ones_like(self.o) * (1.0 / self.minicolumns)
        self.z_co = np.ones((self.n_units, self.n_units)) * (1.0 / self.minicolumns ** 2)

        self.p_pre = np.zeros_like(self.o)
        self.p_post = np.zeros_like(self.o)
        self.p_co = np.zeros((self.n_units, self.n_units))

        self.a = np.zeros_like(self.o)

        if not keep_connectivity:
            self.beta = np.log(np.ones_like(self.o) * (1.0 / self.minicolumns))
            self.w = np.zeros((self.n_units, self.n_units))

    def randomize_pattern(self):
        self.o = self.prng.rand(self.n_units)
        self.s = np.log(self.prng.rand(self.n_units))

        # A follows, if o is randomized sent a to zero.
        self.a = np.zeros_like(self.o)

    def update_discrete(self, N=1):
        for n in range(N):
            self.s = self.beta + np.dot(self.w, self.o)
            self.o = softmax(self.s, t=(1/self.G), minicolumns=self.minicolumns)

    def update_continuous(self, dt=1.0, sigma=None):

        if sigma is None:
            sigma = self.prng.normal(0, self.sigma, self.n_units)

        # Updated the probability and the support
        self.s += (dt / self.tau_m) * (self.g_beta * self.beta + self.g_w * np.dot(self.w, self.o)
                                       + self.g_I * log_epsilon(self.I) - self.s - self.g_a * self.a
                                       + sigma)  # This last term is the noise
        # Softmax
        self.o = softmax(self.s, t=(1/self.G), minicolumns=self.minicolumns)

        # Update the adaptation
        self.a += (dt / self.tau_a) * (self.o - self.a)

        # Updated the z-traces
        self.z_pre += (dt / self.tau_z_pre) * (self.o - self.z_pre)
        self.z_post += (dt / self.tau_z_post) * (self.o - self.z_post)
        self.z_co = np.outer(self.z_post, self.z_pre)

        if self.k > epsilon:
            # Updated the probability
            self.p_pre += (dt / self.tau_p) * (self.z_pre - self.p_pre) * self.k
            self.p_post += (dt / self.tau_p) * (self.z_post - self.p_post) * self.k
            self.p_co += (dt / self.tau_p) * (self.z_co - self.p_co) * self.k

            self.w = get_w_pre_post(self.p_co, self.p_pre, self.p_post)
            self.beta = get_beta(self.p_post)


class NetworkManager:
    """
    This class will run the BCPNN Network. Everything from running, saving and calcualting quantities should be
    methods in this class.  In short this will do the running of the network, the learning protocols, etcera.

    Note that data analysis should be conducted into another class preferably.
    """

    def __init__(self, nn=None, dt=0.001, T_training=1.0, T_ground=1.0, T_recalling=10.0, repetitions=1.0,
                 resting_state=False, values_to_save=[]):
        """
        :param nn: A BCPNN instance
        :param time: A numpy array with the time to run
        :param values_to_save: a list with the values as strings of the state variables that should be saved
        """

        self.nn = nn

        # Timing variables
        self.dt = dt
        self.T_training = T_training
        self.T_ground = T_ground
        self.repetitions = repetitions
        self.resting_state = resting_state

        self.T_recalling = T_recalling

        self.time_training = np.arange(0, self.T_training, self.dt)
        self.time_ground = np.arange(0, self.T_ground, self.dt)
        self.time_recalling = np.arange(0, self.T_recalling, self.dt)

        self.sampling_rate = 1.0

        # Initialize saving dictionary
        self.saving_dictionary = None
        self.update_saving_dictionary(values_to_save)

        # Initialize the history dictionary for saving values
        self.history = None
        self.empty_history()

        # Trained patterns
        self.n_patterns = None
        self.patterns = None

    def calculate_total_training_time(self):
        # The network needs to be trained before

        if self.n_patterns is None:
            raise NameError('The network needs to be trained before')

        if self.resting_state:
            T_total = self.n_patterns * self.repetitions * (self.T_training + self.T_ground)
        else:
            T_total = self.n_patterns * self.repetitions * self.T_training

        return T_total

    def update_saving_dictionary(self, values_to_save):
        """
        This resets the saving dictionary and only activates the values in values_to_save
        """

        # Reinitialize the dictionary
        self.saving_dictionary = {'o': False, 's': False, 'z_pre': False,
                                  'z_post': False, 'z_co': False, 'a': False,
                                  'p_pre': False, 'p_post': False, 'p_co': False,
                                  'w': False, 'beta': False}

        # Activate the values passed to the function
        for state_variable in values_to_save:
            self.saving_dictionary[state_variable] = True

    def empty_history(self):
        """
        A function to empty the history
        """
        empty_array = np.array([]).reshape(0, self.nn.n_units)
        empty_array_square = np.array([]).reshape(0, self.nn.n_units, self.nn.n_units)

        self.history = {'o': empty_array, 's': empty_array, 'z_pre': empty_array,
                        'z_post': empty_array, 'z_co': empty_array_square, 'a': empty_array, 'p_pre': empty_array,
                        'p_post': empty_array, 'p_co': empty_array_square, 'w': empty_array_square, 'beta': empty_array}


    def run_network(self, time=None, I=None):
        # Change the time if given
        if time is not None:
            self.time = time
        # IPython.embed()
        self.dt = time[1] - time[0]

        # Load the clamping if available
        if I is None:
            self.nn.I = np.zeros_like(self.nn.o)
        else:
            self.nn.I = I

        # Create a vector of noise
        noise = self.nn.prng.normal(0, self.nn.sigma, size=(self.time.size, self.nn.n_units))

        # Initialize run history
        run_history = {}

        for quantity, boolean in self.saving_dictionary.items():
            if boolean:
                run_history[quantity] = []

        # Run the simulation and save the values
        for index_t, t in enumerate(self.time):
            if self.saving_dictionary['o']:
                run_history['o'].append(np.copy(self.nn.o))
            if self.saving_dictionary['s']:
                run_history['s'].append(np.copy(self.nn.s))
            if self.saving_dictionary['z_pre']:
                run_history['z_pre'].append(np.copy(self.nn.z_pre))
            if self.saving_dictionary['z_post']:
                run_history['z_post'].append(np.copy(self.nn.z_post))
            if self.saving_dictionary['z_co']:
                run_history['z_co'].append(np.copy(self.nn.z_co))
            if self.saving_dictionary['a']:
                run_history['a'].append(np.copy(self.nn.a))
            if self.saving_dictionary['p_pre']:
                run_history['p_pre'].append(np.copy(self.nn.p_pre))
            if self.saving_dictionary['p_post']:
                run_history['p_post'].append(np.copy(self.nn.p_post))
            if self.saving_dictionary['p_co']:
                run_history['p_co'].append(np.copy(self.nn.p_co))
            if self.saving_dictionary['w']:
                run_history['w'].append(np.copy(self.nn.w))
            if self.saving_dictionary['beta']:
                run_history['beta'].append(np.copy(self.nn.beta))

            # Update the system
            self.nn.update_continuous(dt=self.dt, sigma=noise[index_t, :])

        # Concatenate with the past and redefine dictionary
        for quantity, boolean in self.saving_dictionary.items():
            if boolean:
                self.history[quantity] = np.concatenate((self.history[quantity], run_history[quantity]))

        return self.history

    def run_network_training(self, patterns, repetitions=None, resting_state=None):
        """
        This runs a network training protocol
        :param patterns: a sequence of patterns passed as a list
        :param repetitions: the number of time that the sequence of patterns is trained
        :param resting_state: whether there will be a resting state between each sequence
        :return:
        """

        self.patterns = patterns
        self.n_patterns = len(patterns)

        if repetitions is None:
            repetitions = self.repetitions
        if resting_state is None:
            resting_state = self.resting_state

        for i in range(repetitions):
            print('repetitions', i)
            for pattern in patterns:
                self.nn.k = 1.0
                self.run_network(time=self.time_training, I=pattern)
                self.nn.k = 0.0
                if resting_state:
                    self.run_network(time=self.time_ground)

    def run_network_recall(self, reset=True, empty_history=True):
        """
        Run network free recall
        :param reset: Whether the state variables values should be returned
        :param empty_history: whether the history should be cleaned
        """

        if empty_history:
            self.empty_history()
        if reset:
            self.nn.reset_values(keep_connectivity=True)

        self.run_network(time=self.time_recalling)
