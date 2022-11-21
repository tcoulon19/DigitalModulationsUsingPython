import numpy as np
import abc
import matplotlib.pyplot as plt

class Modem:

    __medadata__ = abc.ABCMeta

    # Base class: Modem
    # Attribute definitions:
    #   self.M: number of points in the MPSK constellation
    #   self.name: name of modem: PSK, QAM, PAM, FSK
    #   self.constellation: reference constellation
    #   self.coherence: only for 'coherent' or 'noncoherent' FSK

    def __init__(self,M,constellation,name,coherence=None): # Constructor

        if (M<2) or ((M & (M-1))!=0): # If M not a power of 2
            raise ValueError('M should be a power of 2')
        if name.lower() == 'fsk':
            if (coherence.lower()=='coherent') or (coherence.lower()=='noncoherent'):
                self.coherence = coherence
            else:
                raise ValueError('Coherence must be \'coherent\' or \'noncoherent\'')
        else:
            self.coherence = None
        
        self.M = M # Number of points in the constellation
        self.name = name # Name of the modem: PSK, QAM, PAM, FSK
        self.constellation = constellation # Reference constellation

    def plotConstellation(self):

        '''
        Plot the reference constellation points for the selected modem
        '''
        if self.name.lower() == 'fsk':
            return 0 # FSK is multi-dimenstional, difficult to visualize

        

    