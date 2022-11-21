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

        plt.figure(0)
        fig, axs = plt.subplots(1,1)
        axs.plot(np.real(self.constellation),np.imag(self.constellation),'o')

        for i in range(0,self.M):
            axs.annotate("{0:0{1}b}".format(i,int(np.sqrt(self.M))), (np.real(self.constellation[i]), np.imag(self.constellation[i])))

        axs.set_title('Constellaton')
        axs.set_xlabel('I')
        axs.set_ylabel('Q')
        fig.savefig('Ch3_images/plotConstellation.png')

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
        Note: MPAM/BPSK are one dimensional modulations. The same function can be
        applied for these modulations since quadrature is zero (Q=0)

        The function computes the pair-wise Euclidean distance of each point in the 
        received vector against every point in the reference constellation. It then
        returns the symbols from the reference constellation that provide the 
        minimum Euclidean distance.

        Parameters:
            receivedSyms: received symbol vector of complex form
        Returns:
            detectedSyms: decoded symbols that provide minimum Euclidean distance
        '''
        from scipy.spatial.distance import cdist

        # Received vector and reference in cartesian form
        XA = np.column_stack((np.real(receivedSyms),np.imag(receivedSyms)))
        XB = np.column_stack((np.real(self.constellation),np.imag(self.constellation)))

        d = cdist(XA, XB, metric='euclidean') # Compute pair-wise Euclidean distances
        detectedSyms = np.argmin(d,axis=1) # Indices corresponding minimum Euclidean distance
        return detectedSyms

         
class PAMModem(Modem):

    # Derived class: PAMModem
    def __init__(self,M):

        m = np.arange(0,M) # All information symbols m = {0,1,...,M-1}
        constellation = 2*m+1-M + 1j*0 # Reference constellation
        Modem.__init__(self, M, constellation, name='PAM') # Set the modem attributes


class PSKModem(Modem):

    # Derived class: PSKModem
    def __init__(self,M):

        # Generate reference constellation
        m = np.arange(0,M) # All information symbols m = {0,1,...,M-1}
        I = 1/np.sqrt(2)*np.cos(m/M*2*np.pi)
        Q = 1/np.sqrt(2)*np.sin(m/M*2*np.pi)
        constellation = I + 1j*Q # Reference constellation
        Modem.__init__(self, M, constellation, name = 'PSK') # Set the modem attributes


class QAMModem(Modem):

    # Derived class: QAMModem
    def __init__(self,M):
        
        if (M==1) or (np.mod(np.log2(M),2)!=0): # M not an even power of 2
            raise ValueError('Only square MQAM supported. M must be even power of 2')

        n = np.arange(0,M) # Sequential address from 0 to M-1 (1xM dimension)
        a = np.asarray([x^(x>>1) for x in n]) # Convert linear addresses to Gray code
        D = np.sqrt(M).astype(int) # Dimension of K-Map - NxN matrix
        a = np.reshape(a,(D,D)) # NxN gray coded matrix
        oddRows = np.arange(1,D,2) # Identify alternate rows
        a[oddRows,:] = np.fliplr(a[oddRows,:]) # Flip rows - KMap representation
        nGray = np.reshape(a,(M)) # Reshape to 1xM - Gray code walk on KMap

        # Construction of ideal M-QAM constellation from sqrt(M)-PAM
        (x,y) = np.divmod(nGray,D) # Element-wise quotient and remainder
        Ax = 2*x+1-D # PAM Amplitudes 2d+1-D - real axis
        Ay = 2*y+1-D # PAM Amplitudes 2d+1-D - imag axis
        constellation = Ax+1j*Ay
        Modem.__init__(self, M, constellation, name = 'QAM') # Set the modem attributes


class FSKModem(Modem):

    # Derived class: FSKModem
    def __init__(self,M,coherence='coherent'):
        
        if coherence.lower()=='coherent':
            phi = np.zeros(M) # Phase=0 for coherent detection
        elif coherence.lower()=='noncoherent':
            phi = 2*np.pi*np.random.rand(M) # M random phases in the (0,2pi)
        else:
            raise ValueError('Coherence must be \'coherent\' or \'noncoherent\'')
        constellation = np.diag(np.ex(1j*phi))
        Modem.__init__(self,M,constellation,name='FSK',coherence=coherence.lower()) # Set the base modem attributes

    def demodulate(self, receivedSyms, coherence = 'coherent'):

        # Overridden method in Modem class
        if coherence.lower() == 'coherent':
            return self.iqDetector(receivedSyms)
        elif coherence.lower() == 'noncoherent':
            return np.argmax(np.abs(receivedSyms),axis=1)
        else:
            raise ValueError('Coherence must be \'coherent\' or \'noncoherent\'')






    