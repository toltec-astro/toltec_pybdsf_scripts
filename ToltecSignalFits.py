from scipy.interpolate import RectBivariateSpline as rbs
from matplotlib import pyplot as plt
from astropy.nddata import Cutout2D
from astropy import units as u
import scipy.constants as sc
from astropy.io import fits
from astropy.wcs import WCS
from BdsfCat import BdsfCat
import numpy as np
import lmfit
import glob
plt.ion()

# A generic class to easily grab the contents of a TolTEC signal filts
# image as output from Citlali.

class ToltecSignalFits:
    """This is a helper class to support reading and doing common operations
       with TolTEC output fits files from citlali.
       Objects are instantiated by specifying the path to the fits file and
       the name of the array ('a1100', 'a1400', or 'a2000').

       Note that this class *only* reads the metadata from the files and not
       the actual images unless its member functions are explicitly called.
    """

    #instantiate
    def __init__(self, path='.', array='a1100'):
        """Instantiator for a ToltecSignalFits object.
        Inputs: 
          path (string) - the path to the output FITS files.
          array (string) - one of 'a1100', 'a1400', or 'a2000'
        """

        # check the input array
        if(array not in ['a1100', 'a1400', 'a2000']):
            raise ValueError(
                "Input array string must be one of 'a1100', 'a1400', 'a2000'.")

        # define some useful items we just happen to know
        if(array == 'a1100'):
            self.freq = sc.c/(1.1e-3)*u.Hz
        elif(array == 'a1400'):
            self.freq = sc.c/(1.4e-3)*u.Hz
        else:
            self.freq = sc.c/(2.e-3)*u.Hz
        
        # find the corresponding fits file
        # don't include noise maps
        ffile = glob.glob(path+'*{}*.fits'.format(array))
        self.filename = [i for i in ffile if 'noise' not in i]
        if(len(ffile) == 0):
            raise OSError('No fits file for {} found at {}'.format(array, path))
        self.filename = self.filename[0]
        print("Fits file found: "+self.filename)
        
        # open it and extract the key metadata
        with fits.open(self.filename, mmap=True) as hd:
            self.nExtensions = len(hd)
            self.extNames = ['None']
            for h in hd[1:]:
                self.extNames.append(h.header['EXTNAME'])
            self.headers = []
            for h in hd:
                self.headers.append(h.header)
            
        self.array = self.headers[0]['WAV']
        self.to_mJyPerBeam = self.headers[0]['HIERARCH TO_MJY/B']
        self.obsnum = self.headers[0]['OBSNUM']
        self.units = self.headers[0]['UNIT']

        # set a default weight cut to 0 so we default to getting
        # everything
        self.weightCut = 0
        self.weight = None

        # determine the beam sizes by fitting a gaussian to the kernel
        # map
        r, pos = self.fitImageToGaussian('kernel', verbose=False)
        self.beam = r.params['fwhmx'].value*u.arcsec
        self.kerFunc = None


    def getMap(self, name):
        """Extracts an image from the Fits file.  
           The image is referenced by its extname. Valid input names are:
           'signal', 'weight', 'kernel', 'coverage', 'sig2noise'

           If weightCut is nonzero, pixels in the image with associated weight
           that are less than weightCut are set to zero.
        """
        i = self._checkInputName(name)
        with fits.open(self.filename, mmap=True) as hd:
            image = hd[i].data

        # apply the weight cut if set
        if(self.weightCut > 0.):
            self.getWeight()
            w = np.nonzero(self.weight <
                           self.weightCut*self.weight.max())
            if len(w) > 0:
                image[w] = 0.
        return image
            

    def setWeightCut(self, wc):
        """Sets a fractional weight limit for the output maps.  This is
         useful when you want to trim off the low-weight outer edges
         of the map.
         Input: 0 <= weightCut <= 1"""
        if(wc < 0) | (wc>1):
            p1 = "Input must be between 0 (no cut) and 1 (everything's cut). "
            raise ValueError(p1+"Not {}".format(wc))
        self.weightCut = wc


    def getWeight(self):
        """Read the weight map into memory."""
        if(self.weight is not None):
            return
        i = self.extNames.index('weight')
        with fits.open(self.filename, mmap=True) as hd:
            self.weight = hd[i].data

            
    def plotImage(self, name, image=None,
                  vmin=None, vmax=None, ax=None,
                  units='MJy/sr'):
        """Makes a nice plot of the image using the WCS projection.
        Inputs:
          - name (string), the EXTNAME of the image or WCS.
          - image (array), optional image to use in place of the FITS image
          - vmin, vmax (floats), min and max fluxes for imshow
          - units (string), one of 'MJy/sr' (default) or 'mJy/beam'
        """
        # verify the name and get the index of the hdu
        i = self._checkInputName(name)

        # if the image is not provided as a keyword, fetch it using
        # the name
        if image is None:
            image = self.getMap(name)

        # choose your units
        if(units == 'mJy/beam'):
            if(name == 'weight'):
                image /= (self.to_mJyPerBeam)**2
            if(name == 'signal'):
                image *= self.to_mJyPerBeam                
            else:
                print("Only the signal and weight images can change units.")

        # finally fetch the wcs and make the plot
        wcs = WCS(self.headers[i])
        if(ax is None):
            ax = plt.subplot(projection=wcs)
        ax.imshow(image, origin='lower', vmin=vmin, vmax=vmax)
        ax.grid(color='white', ls='solid')

        # todo: add a colorbar
        return ax

    
    def fitImageToGaussian(self, name, pos=None,
                           size=(20, 20), plot=False,
                           verbose=True):
        """Fits a Gaussian shape to a postage stamp cutout of an image.
        Inputs:
          - name (string), the EXTNAME of the image or corresponding WCS.
          - pos (float,foat), x- and y-pixel positions of the source location
          - size (float, float), x- and y-sizes of the cutout image
          - plot (bool) - make a plot of the cutout and centroid?
          - verbose (bool) - wordy output
        """
        image = self.getMap(name)
        self.getWeight()
        if(pos is None):
            pos = (self.headers[0]['CRPIX1'], self.headers[0]['CRPIX2'])
        cutout = Cutout2D(image, pos, size)
        weightout = Cutout2D(self.weight, pos, size)
        Z = cutout.data
        W = weightout.data
        X, Y = np.mgrid[0:size[0], 0:size[1]]
        # flatten these arrays
        z = Z.flatten()
        x = X.flatten()
        y = Y.flatten()
        w = W.flatten()
        # the 2d gaussian fit
        model = lmfit.models.Gaussian2dModel()
        params = model.guess(z, x, y)
        params['amplitude'].value = z.max()
        result = model.fit(z, x=x, y=y, params=params, weights=w)
        if(verbose):
            lmfit.report_fit(result)
        if(plot):
            plt.figure(2)
            plt.imshow(Z)
            plt.axvline(x=result.params['centery'].value)
            plt.axhline(y=result.params['centerx'].value)
        centerpos = (result.params['centery'].value,
                     result.params['centerx'].value)

        # todo:
        #  - add a colorbar to the plot
        #  - add the results of the fit as a legend or annotation
        
        return result, cutout.to_original_position(centerpos)


    def _constructKernelInterp(self):
        """Builds a 2-d interpolation of the kernel image.  This is 
        useful for source modeling, addition and subtraction."""
        k = self.getMap('kernel')
        r, pos = self.fitImageToGaussian('kernel', verbose=False)
        self.kerFunc = rbs(np.arange(k.shape[0])-pos[1],
                           np.arange(k.shape[1])-pos[0], k)


    def subtractKernelFromImage(self, name, image=None,
                                sourcePos=None,
                                sourceAmp=None, fitForVals=False):
        """Subtract a kernel shaped source from an image.
        Inputs:
         - name (string), the EXTNAME of the image or corresponding WCS.
         - image [array], optional - an input image with same WCS as "name" 
                         from which you want to subtract a kernel-shaped source.
                         If no image is provided, the image corresponding to "name"
                         is used.
         - sourcePos [int x 2] - the source position, in pixels, to be subtracted.
         - sourceAmp [float] - the amplitude of the source to be subtracted.  If no
                               amplitude is provided, the amplitude at the sourcePos
                               position of the image is used.
         - fitForVals [bool] - set to True to have the code fit for the source position
                               and amplitude using pos and amp as guesses if provided.
        """
        # Preliminaries
        if image is None:
            image = self.getMap(name)
        self.getWeight()
        # Deal with the inputs.  If pos is not given, head for center
        # of the map.
        if(sourcePos is None):
            sourcePos = (int(self.headers[0]['CRPIX1']),
                         int(self.headers[0]['CRPIX2']))
        # get pos from a fit, use whatever pos is as a target for the fit
        if(fitForVals == True):
            r, sourcePos = self.fitImageToGaussian(name, pos=sourcePos,
                                                   verbose=False)
            a = r.params['sigmax'].value*r.params['sigmay'].value*2.*np.pi
            sourceAmp = r.params['amplitude']/a
        # if amplitude is not given, take the amplitude at pos
        if(sourceAmp is None):
            sourceAmp = image[(round(sourcePos[1]), round(sourcePos[0]))]

        # get kernel interpolation function
        if(self.kerFunc is None):
            self._constructKernelInterp()
        
        # subtract
        # x = np.arange(image.shape[0])-sourcePos[1]
        # y = np.arange(image.shape[1])-sourcePos[0]
        # kp = self.kerFunc(x, y)
        # kp = kp/kp.max()*sourceAmp
        # return image-kp

        # create a (40x40) cutout for performing the subtraction
        c = Cutout2D(image, (int(sourcePos[0]),int(sourcePos[1])), 100, copy=False)
        sp = c.to_cutout_position(sourcePos)
        x = np.arange(c.shape[0])-sp[1]
        y = np.arange(c.shape[1])-sp[0]
        kp = self.kerFunc(x, y)
        kp = kp/kp.max()*sourceAmp
        c.data -= kp
        return image
        


    def removeBdsfCatalogFromImage(self, name, image=None,
                                   catFile=None,
                                   fluxLimit=0.,
                                   fluxCorr=1.0):
        """Subtract a kernel shaped source from an image.
        Inputs:
         - name (string), the EXTNAME of the image or corresponding WCS.
         - image [array], optional - an input image with same WCS as "name" 
                         from which you want to subtact off the catalog.
                         If no image is provided, the image corresponding to "name"
                         is used.
         - catFile [string] - a PyBDSF catalog FITS file.
         - fluxLimit [float] - the flux limit of sources to subtract.  No sources with
                               catalog fluxes less than this limit will be subtracted.
         - fluxCorr [float] - a user-supplied flux correction factor to apply to the
                              catalog fluxes prior to subtraction.

        """
        # deal with the image and wcs first
        i = self._checkInputName(name)
        if image is None:
            image = self.getMap(name)
            image *= self.to_mJyPerBeam
        wcs = WCS(self.headers[i])
        
        # Set the minumum flux for a removed source to be 3-sigma
        self.getWeight()
        sigma = 1./np.sqrt(self.weight.max()) * self.to_mJyPerBeam
        fluxLimit = max(fluxLimit, sigma*1.)

        # Read in the catalog
        c = BdsfCat(catFile, verbose=True)
        nSources = len(c.sources)
        
        # Subtract the sources
        i = 0
        for s in c.sources:
            if(s.peakFlux >= fluxLimit):
                i += 1
                px, py = wcs.world_to_pixel(s.coords)
                pos = (px.min(), py.min())
                print()
                print('Subtracting Source: ')
                print('  amp={}'.format(s.peakFlux))
                print('  corrected amp={}'.format(s.peakFlux*fluxCorr))
                print('  pos={}'.format(pos))
                print('  Image flux near pos: {}'.format(image[(round(pos[1]), round(pos[0]))]))
                image = self.subtractKernelFromImage(name, image=image,
                                                     sourcePos=pos,
                                                     sourceAmp=s.peakFlux*fluxCorr)
        nSubtracted = i
        self.bdsf_catalog_nSources = nSources
        self.bdsf_catalog_nSubtracted = nSubtracted
        return image
        
        

    def addKernelToImage(self, name, image=None,
                         sourcePos=None, sourceAmp=1.0):
        """Add a kernel shaped source from an image.
        Inputs:
         - name (string), the EXTNAME of the image or corresponding WCS.
         - image [array], optional - an input image with same WCS as "name" 
                         from which you want to add the source.
                         If no image is provided, the image corresponding to "name"
                         is used.
         - sourcePos [floats] - source locations (x, y) in pixel coordinates.  If None,
                                then the center of the image is used.
         - sourceAmp [float] - the amplitude of the kernel shape source to be added.
        """
        
        # Preliminaries
        if image is None:
            image = self.getMap(name)
        # Deal with the inputs.  If pos is not given, head for center
        # of the map.
        if(sourcePos is None):
            sourcePos = (int(self.headers[0]['CRPIX1']),
                         int(self.headers[0]['CRPIX2']))
        # get kernel interpolation function
        if(self.kerFunc is None):
            self._constructKernelInterp()
        # add kernel shaped source
        x = np.arange(image.shape[0])-sourcePos[1]
        y = np.arange(image.shape[1])-sourcePos[0]
        kp = self.kerFunc(x, y)
        kp = kp/kp.max()*sourceAmp
        return image+kp

    
    def writeImageToFits(self, name, fitsfile, JyPerBeam=False,
                         rmsMeanMap=None, overwrite=False):
        """Reads in the requested signal from the fits file and writes it
           as the primary extension in a new fitsfile."""
        i = self._checkInputName(name)
        image = self.getMap(name)
        header = self.headers[i]

        if(JyPerBeam):
            image *= self.to_mJyPerBeam
            image *= 0.001
            avgNoise = 1./np.sqrt(self.weight.max())*self.to_mJyPerBeam*0.001
            print()
            print("Map UNITS now Jy/Beam")
            print("Average noise in map is {} Jy/Beam.".format(avgNoise))
            print()
        else:
            avgNoise = 1./np.sqrt(self.weight.max())*self.to_mJyPerBeam

        header.append(('BMAJ', self.beam.to_value(u.deg)))
        header.append(('BMIN', self.beam.to_value(u.deg)))
        header.append(('BPA', 0.))
        header.append(('CDELT1', header['CD1_1']))
        header.append(('CDELT2', header['CD2_2']))
        header.append(('FREQ', self.freq.to_value(u.Hz)))
        hduP = fits.PrimaryHDU(image, header)
        hdulist = fits.HDUList([hduP])
        hdulist.writeto(fitsfile, overwrite=overwrite)

        if(rmsMeanMap is not None):
            rmsMapName = rmsMeanMap[0]
            meanMapName = rmsMeanMap[1]
            rms = np.zeros(image.shape)
            w = np.nonzero(self.weight >=
                           self.weightCut*self.weight.max())
            rms[w] = 1./np.sqrt(self.weight[w])*self.to_mJyPerBeam*0.001
            hduP = fits.PrimaryHDU(rms, header)
            hdulist = fits.HDUList([hduP])
            hdulist.writeto(rmsMapName, overwrite=overwrite)

            # need a map of zeros as well
            zeros = np.zeros(image.shape)
            hduP = fits.PrimaryHDU(zeros, header)
            hdulist = fits.HDUList([hduP])
            hdulist.writeto(meanMapName, overwrite=overwrite)
        return avgNoise


    def _checkInputName(self, name):
        try:
            i = self.extNames.index(name)
        except:
            raise ValueError('Input name must be in {}'.format(self.extNames))
        return i
