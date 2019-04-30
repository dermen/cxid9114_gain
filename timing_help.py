
from scipy.signal import correlate2d 
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline

try: 
    import Tkinter as tk
except ImportError:
    import tkinter as tk
try:
    import psana
    has_psana = True
except ImportError:
    has_psana=False
import numpy as np
import widgets

class PsanaImages:
    def __init__(self, exp, run, detector_name, codes=None):
        """
        exp: experiment string
        run: run number
        detector_name: detecrtor string
        """
        assert( has_psana)
       
        self.run_str = run
        self.exp = exp
        self.codes = codes
        self.ds = psana.DataSource("exp=%s:run=%d:idx"%(exp,run))
        
        self.run = self.ds.runs().next()
        self.times = self.run.times()
        self.N_events = self.N = len( self.times)
        self.event_info_string = "Start" 

        self.detnames = [ d for sl in psana.DetNames() for d in sl]
        self.env = self.ds.env()
        
        assert(  detector_name  in self.detnames)
       
        self.code_dets = [ psana.Detector(d, self.env) 
                    for d in self.detnames if d.startswith("evr") ]
        
        self.detector_name = detector_name
        self.Detector = psana.Detector( self.detector_name, self.env)
        self.gas_reader = psana.Detector("FEEGasDetEnergy", self.env)
        self.spectrometer = psana.Detector("FeeSpec-bin", self.env)

#       Get image shape..
        I = None
        i = 0
        while I is None:
            ev = self.run.event(self.times[i])
            if ev is None:
                i += 1
                continue
            I = self.Detector.image(ev)
            i += 1
        self.img_sh = I.shape
        self.empty_img = np.zeros( self.img_sh)
        self.gain_map = self.Detector.gain_mask(run)>1

    def __getitem__(self, i):
        self.event_index = i
        self.event = self.run.event( self.times[i] )

        img = self._get_image()
        return img

    def _get_image( self):
        if self.event is None:
            self.shot_i = self.event_index
            self.N_i = self.N_events
            self.event_info_string = "Broken event"
            img = self.empty_img
        else:
            self.codes = []
            for cdet in self.code_dets:
                c = cdet.eventCodes(self.event)
                if c is not None:
                    self.codes += c
            self.event_codes = list(set(self.codes))
            self.evr_str = " ".join([str(c) for c in self.event_codes ])
            self.event_info_string = "run: %d; exp: %s; evr: %s"\
                    %(self.run_str, self.exp, self.evr_str )
            self.shot_i = self.event_index
            self.N_i = self.N_events

            img = self.Detector.calib(self.event)
            gas_reading = self.gas_reader.get( self.event)
            spec = self.spectrometer.image(self.event)
            if gas_reading is None:
                gas_str = "None"
            else:
                gas_str = "%.9f" % gas_reading.f_11_ENRC()
            if img is None:
                self.event_info_string = "Nonetype image, gas: %s" % gas_str
                img = self.empty_img
            else:
                self.event_info_string = "run: %d; exp: %s; evr: %s, gas: %s"\
                    %(self.run_str, self.exp, self.evr_str, gas_str )
            if spec is None:
                spec_line = np.zeros(1024)
                spec_img = np.zeros( (256,1024))
            else:
                spec_out = get_spectrum(spec)
                spec_line = spec_out[-1]
                spec_img = spec_out[-2]
                print spec_img.shape
        return img, spec_line, spec_img
    

def ellipse_structure(a,b):
    a = float(a)
    b = float(b)
    size = max(a,b)*2+1
    i,j = np.mgrid[0:size, 0:size]
    i -= (size/2)
    j -= (size/2)
    return np.sqrt(i**2+ a**2 * j**2/ b**2) <= a


def get_spectrum(spec):

    Efit = np.array([1.45585875e-01, 8.92420032e+03])
    # (use with polyval) 
    
    bg_rows = 20  # number of rows in image where we expect only background
    nsig = 5  # how many 
    ev_width=5
    V = 22  # height of ellipse matcher
    H = 9  # width of ellipse matcher 
    
    # template match on the camera
    foot = ellipse_structure(V/2,H/2)  # ellipse footprint
    matcher = correlate2d( spec, foot, mode='same') / np.sum( foot)
    bg_pixels = np.hstack( (matcher[:bg_rows].ravel(), 
                            matcher[-bg_rows:].ravel() ))

    pfit = np.array([-7.29833699e-02,  1.59887735e+02])  #  use a predetermined fit.. 

    # lines to interpolate along
    nn=V*3  # how many lines
    xdata = np.arange(spec.shape[1])
    ydatas = []
    for i in np.arange(-nn,nn+1,1):
        ydatas.append( np.polyval( pfit + np.array([0,i]), xdata ) ) 

    # interpolate
    rbs = RectBivariateSpline( np.arange(spec.shape[0]), 
                            np.arange(spec.shape[1]), matcher)
    evals4 = []  # evaluations
    for y in ydatas:
        evals4.append( rbs.ev(y, xdata))

    bg_sig = bg_pixels.std()
    bg_mean = bg_pixels.mean() 
    evals4 = np.array( evals4)
    evals4_ma = np.ma.masked_where(evals4 < bg_mean+bg_sig*nsig , evals4)
    raw_spec = np.sum( evals4_ma,axis=0)

    Edata = np.polyval( Efit,xdata)
    en_bins = np.arange( Edata[0],Edata[-1]+1, ev_width)

#   now we can bin 
    spec_hist = np.histogram( Edata , en_bins, weights=raw_spec)[0]

    return en_bins[1:]*.5  + en_bins[:-1]*.5, spec_hist , matcher, raw_spec

