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

        fig, axs = plt.subplots(1,1)
        axs.plot(np.real(self.constellation),np.imag(self.constellation),'o')

        for i in range(0,self.M):
            axs.annotate("{0:0{1}b}".format(i,self.M), (np.real(self.constellation[i]), np.imag(self.constellation[i])))

        axs.set_title('Constellaton')
        axs.set_xlabel('I')
        axs.set_ylabel('Q')
        fig.show()

    def modulate(self,inputSymbols):

        '''
        Modulate a vector of input symbols (numpy array format) using the chosen modem.
        Input symbols take integer values in the range 0 to M-1.
        '''
        if isinstance(inputSymbols, list):
            inputSymbols = np.array(inputSymbols)

        if not (0 <= inputSymbols.all() <= self.M-1):
            raise ValueError('inputSymbols values are beyond the range 0 to M-1')

        modulatedVec = self.constellation[inputSymbols]
        return modulatedVec # Return modulated vector

    def demodulate(self,receivedSyms):
        
        '''
        Demodulate a vector of received symbols using the chosen modem.
        '''
        if isinstance(receivedSyms,list):
            receivedSyms = np.array(receivedSyms)

        detectedSyms = self.iqDetector(receivedSyms)
        return detectedSyms

    def iqDetector(self,receivedSyms):
        '''
        Optimum Detector for 2-dim. signals (ex: MQAM,MPSK,MPAM) in IQ plane
        '''

    