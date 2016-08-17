import numpy as np


class BCPNN:
    def __init__(self,beta, w, s=None, o=None, a=None, z_pre=None, z_post=None,
                 p_pre=None, p_post=None, p_co=None):

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


    def update_discrete(self):
        


    def update_continuous(self):

