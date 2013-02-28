import unittest
import numpy as np
from math import sqrt
## Truth information


class TruthParticle(object):
    """
    A TruthParticle represents a physics object defined by its 4-vector along with quantum numbers.
    It further defines where it was created, and relations to it parent and children (sibling relations?).
    A point of decay can be defined as well
    
    """
    def __init__(self, pdgId=None, x0=None, p0=None, q=None, spin=None):
        super(TruthParticle, self).__init__()
        
        self.pdgId = pdgId
        self.x = x0 # [x,y,z,t]
        self.p = p0 # [px, py, pz, E]
        self.charge = q
        self.spin = None
        self.color = None

        self.decay_point = None # [z,y,z,t]
        
        self.children = []
        self.parents = []
        
    def siblings(self):
        """Return Siblings"""
        if len(self.parents) > 0:
            return [particle for particle in self.parents[0].children if not particle is self]


    def mass(self):
        """docstring for mass"""
        return sqrt(self.p[3]**2 - (self.p[0]**2 + self.p[1]**2 + self.p[2]**2))


################## Experimental sizes #####################################
class RecoTrack(object):
    """A RecoTrack is a reconstructed particle with detector track and hits matching a truth track potentially"""
    def __init__(self):
        super(RecoTrack, self).__init__()
        self.r = None
        self.z = None
        self.P = None # Covariance matrix
        self.Q = None # Process noise
        self.hits_on_track = [] # Hits defining the track

class TruthTrack(object):
    """docstring for Track"""
    def __init__(self):
        super(TruthTrack, self).__init__()
        self.measurements = []

        
class Detector(object):
    """A Detector is a physical representation with geometry, resolution and uncertainties"""
    def __init__(self):
        super(Detector, self).__init__()
        self.hits = []
        self.name = ""
        

class Measurement(object):
    """docstring for Measurement"""
    def __init__(self, detector):
        super(Measurement, self).__init__()
        self.detector = detector
        self.position = None # [x,y,z,]
        self.true_particle = None
        self.true_track = None
        self.charge = None
        self.time = None


class Event(object):
    """docstring for Event"""
    def __init__(self, title = "Event"):
        super(Event, self).__init__()
        self.title = title

        # Truth information
        self.true_particles = []

        # Detector and Experimental information
        self.detectors = {}
        self.detector_fz = [] # detector planes as a function of z

        # Hit information
        self.hits = []
        # Reconstructed information
        
        

# Unit tests below
#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#---#



class TruthParticleTest(unittest.TestCase):
    def setUp(self):
        """docstring for setUp"""
        pass
        
    def test_truth_graph(self):
        """docstring for particle_graph"""
        
        proton1 = TruthParticle(pdgId=2212, x0=np.array([0, 0, -10, 0]), p0=np.array([0.0, 0.0, 23.0, 23.9]), q=1, spin=0.5 )
        proton2 = TruthParticle(pdgId=2212, x0=np.array([0, 0, 10, 0]), p0=np.array([0.0, 0.0, -23.0, 23.9]), q=1, spin=0.5 )
        
        pp_scattering_point = np.array([0, 0, 0, 1.0e-7])

        pp_E_cm = sqrt((proton1.p[3]+proton2.p[3])**2 -  (sum((proton1.p + proton2.p)[0:3]))**2)

        for i in xrange(8): # Create 8 pions
            pion = TruthParticle(pdgId=211, x0=pp_scattering_point, p0=[0.0, 4.0, 0.0, 23.9], q=1, spin=0.5 )
            pion.parents.append(proton1)
            pion.parents.append(proton2)
            proton1.children.append(pion)
            proton2.children.append(pion)
            
        
        first_pion_siblings = proton1.children[0].siblings()
        if proton1.children[0] in first_pion_siblings: print "here"
        
        return self.assertTrue(True)

if __name__ == '__main__':
	unittest.main()