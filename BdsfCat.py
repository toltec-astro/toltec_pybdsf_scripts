from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u
import numpy as np
import os

class BdsfSource:
    """A container for PyBDSF source catalogs of the "srl" variety."""
    def __init__(self, sId, ra, dec, peakFlux, peakFluxErr):
        self.sId = sId
        self.ra = ra*u.deg
        self.dec = dec*u.deg
        self.peakFlux = peakFlux
        self.peakFluxErr = peakFluxErr
        self.s2n = peakFlux/peakFluxErr
        self.coords = SkyCoord(self.ra, self.dec)


class BdsfCat:
    """Utility class for reading and working with PyBDSF FITS catalogs."""
    def __init__(self, catFile, array='a1100', verbose=False):
        """
        Creating a BdsfCat object will read in the pyBdsf data, sort the
        catalog in decreasing order by flux, and organize the source 
        candidates into both arrays and BdsfSource objects.
        Inputs:
           catFile (string) - path and name of PyBDSF FITS catalog output.
           array (string) - one of 'a1100' (default), 'a1400', or 'a2000'
        """
        # check that the catalog file exists
        if not os.path.exists(catFile):
            raise ValueError("{} does not exist.".format(catFile))
        self.catFile = catFile
        self.verbose = verbose
        self.array = array
        self.readCat()
        self.sortByFlux()
        self.fluxes = [s.peakFlux for s in self.sources]
        self.ras = [s.ra for s in self.sources]
        self.decs = [s.dec for s in self.sources]
        self.s2ns = [s.s2n for s in self.sources]
        self.coords = SkyCoord([s.coords for s in self.sources])


    def readCat(self):
        """
        Reads the input catalog.  Source fluxes are converted from 
        Jy/beam to mJy/beam.
        """
        self.sources = []
        hdul = fits.open(self.catFile)
        dd = hdul[1].data
        mf = 0.
        for d in dd:
            self.sources.append(BdsfSource(d[0],
                                           d[2],
                                           d[4],
                                           d[8]*1.e3,
                                           d[9]*1.e3))
            if(d[8] > mf):
                mf = d[8]
        mf *= 1.e3
        if(self.verbose):
            print("BDSF catalog summary:")
            print("  Number of sources: {}".format(len(dd)))
            print("  Max source flux: {} mJy".format(mf))
        hdul.close()


    def sortByFlux(self):
        sorted = []
        f = [-s.peakFlux for s in self.sources]
        w = np.argsort(f)
        s = [self.sources[i] for i in w]
        self.sources = s


    def trimToWeight(self, weight, wcs):
        """
        This method trims the BdsfCatalog to only include sources where the 
        input weight map is nonzero.
        Inputs:
          weight (array) - a weight map image for the observation.
          wcs - the corresponding wcs for the input weight map.
        """
        swc = []
        for s in self.sources:
            px, py = wcs.world_to_pixel(s.coords)
            pos = (px.min(), py.min())
            if(weight[(round(pos[1]), round(pos[0]))] > 0):
                swc.append(s)
        self.sources = swc
        self.fluxes = [s.peakFlux for s in self.sources]
        self.ras = [s.ra for s in self.sources]
        self.decs = [s.dec for s in self.sources]
        self.s2ns = [s.s2n for s in self.sources]
        self.coords = SkyCoord([s.coords for s in self.sources])
