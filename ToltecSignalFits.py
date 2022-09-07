from scipy.interpolate import RectBivariateSpline as rbs
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from astropy.nddata import Cutout2D
from astropy import units as u
import scipy.constants as sc
from astropy.io import fits
from astropy.wcs import WCS
from BdsfCat import BdsfCat
import numpy as np
import lmfit
import glob
from astropy.coordinates import SkyCoord  
from astropy.coordinates import ICRS
from astropy.table import Table
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.utils.exceptions import AstropyUserWarning
from astropy.convolution import Gaussian2DKernel
from astropy.modeling import models, fitting
from astropy.convolution import convolve
import warnings
from photutils.psf import IntegratedGaussianPRF, DAOGroup, BasicPSFPhotometry
from photutils.background import MMMBackground
from photutils.aperture import aperture_photometry, CircularAperture, CircularAnnulus, ApertureStats
from photutils.segmentation import detect_sources
from photutils.segmentation import deblend_sources
from photutils.segmentation import SourceCatalog
from photutils.background import Background2D, MedianBackground
from AstropyTab import AstropyTab
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
    def __init__(self, path='.', array='a1100',label=''):
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
        sstring = path+'*{}*.fits'.format(array)
        ffile = glob.glob(sstring)
        print(ffile)
        self.filename = [i for i in ffile if 'noise' not in i]
        if(len(ffile) == 0):
            print("search string is: {}".format(sstring))
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
        self.to_mJyPerBeam = self.headers[0]['HIERARCH TO_MJY/BEAM']
        self.to_MJyPerSr = self.headers[0]['HIERARCH TO_MJY/SR']
        self.obsnum = self.headers[0]['OBSNUM']
        self.citlali_version = self.headers[0]['CREATOR']

        # set a default weight cut to 0 so we default to getting
        # everything
        self.weightCut = 0
        self.weight = None

        # determine the beam sizes by fitting a gaussian to the kernel
        # map
#        r, pos = self.fitImageToGaussian('kernel_I', verbose=False)
        fitker=self.fitGaussian('kernel_I')
        self.beam = ((fitker[0].x_stddev.value+fitker[0].y_stddev.value)*0.5*u.arcmin).to(u.arcsec)
        self.maxKer=fitker[0].amplitude.value
#        self.kerFunc = None
        self.label=label

    def getMap(self, name):
        """Extracts an image from the Fits file.  
           The image is referenced by its extname. Valid input names are:
           'signal_I', 'weight_I', 'kernel_I', 'coverage_I', 'sig2noise_I'

           If weightCut is nonzero, pixels in the image with associated weight
           that are less than weightCut are set to zero.
        """
        i = self._checkInputName(name)
        with fits.open(self.filename, mmap=True) as hd:
            wcsobj = WCS(hd[i].header).sub(2) 
            image = hd[i].data.reshape(wcsobj.array_shape)

        # apply the weight cut if set
        if(self.weightCut > 0.):
            self.getWeight()
            w = np.nonzero(self.weight <
                           self.weightCut*self.weight.max())
            if len(w) > 0:
                image[w] = 0.
        return image
            

    def getMapWCS(self, name):
        """Returns the WCS objects of the map corresponding to "name."
        """
        i = self._checkInputName(name)
        wcsobj = WCS(self.headers[i]).sub(2)
        return wcsobj

        
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
        i = self.extNames.index('weight_I')
        with fits.open(self.filename, mmap=True) as hd:
            wcsobj = WCS(hd[i].header).sub(2) 
            self.weight = hd[i].data.reshape(wcsobj.array_shape)

            
    def plotCutout(self, name, coords, image=None,
                   vmin=None, vmax=None, ax=None,
                   title=None,
                   markerCoords=None,
                   units='mJy/beam',
                   size=(20, 20)):
        """Makes a plot of a cutout of the image using the WCS projection.
        Inputs:
          - name (string), the EXTNAME of the image or WCS.
          - coords (SkyCoord object), the coords of the center of the cutout
          - image (array), optional image to use in place of the FITS image
          - vmin, vmax (floats), min and max fluxes for imshow
          - units (string), one of 'MJy/sr' or 'mJy/beam (default)'
          - title (string), an optional title for the plot
          - markerCoords (SkyCoord object), optional coordinates for a circular marker
          - size ((m,n)), the size in pixels of the cutout (default: (20,20))
        """
        # verify the name and get the index of the hdu
        i = self._checkInputName(name)

        # if the image is not provided as a keyword, fetch it using
        # the name
        if image is None:
            image = self.getMap(name)

        # choose your units
        if(units == 'mJy/beam'):
            converter = self.to_mJyPerBeam
        elif(units == 'MJy/sr'):
            converter = self.to_MJyPerSr
        else:
            raise ValueError("No such unit: {}".format(units))

        if(name == 'weight_I'):
            image /= converter**2
        elif(name == 'signal_I'):
            image *= converter
        else:
            print("Only the signal and weight images can change units.")

        # the wcs
        wcs = WCS(self.headers[i]).sub(2)

        # the cutout
        pos = wcs.world_to_pixel(coords)
        cutout = Cutout2D(image, pos, size, wcs=wcs)

        # the plot
        if(ax is None):
            ax = plt.subplot(projection=cutout.wcs)
        ax.imshow(cutout.data, origin='lower', vmin=vmin, vmax=vmax)
        ax.grid(color='white', ls='solid')
        if(title is not None):
            ax.set_title(title)
        if(markerCoords is not None):
            tx = ax.get_transform('world')
            ax.scatter(markerCoords.ra.value, markerCoords.dec.value,
                       transform=tx, s=100,
                       edgecolor='red', facecolor='none')
        
        return cutout
        
    
    def plotImage(self, name, image=None,
                  vmin=None, vmax=None, ax=None,
                  units='mJy/beam',
                  bdsfCat=None,
                  bdsfCatFluxLow=None,
                  bdsfCatFluxHigh=None):
        """Makes a nice plot of the image using the WCS projection.
        Inputs:
          - name (string), the EXTNAME of the image or WCS.
          - image (array), optional image to use in place of the FITS image
          - vmin, vmax (floats), min and max fluxes for imshow
          - units (string), one of 'MJy/sr' or 'mJy/beam (default)'
        """
        # verify the name and get the index of the hdu
        i = self._checkInputName(name)

        # if the image is not provided as a keyword, fetch it using
        # the name
        if image is None:
            image = self.getMap(name)

        # choose your units
        if(units == 'mJy/beam'):
            converter = self.to_mJyPerBeam
        elif(units == 'MJy/sr'):
            converter = self.to_MJyPerSr
        else:
            raise ValueError("No such unit: {}".format(units))

        if(name == 'weight_I'):
            image /= converter**2
        elif(name == 'signal_I'):
            image *= converter
        else:
            print("Only the signal and weight images can change units.")

        # finally fetch the wcs and make the plot
        wcs = WCS(self.headers[i]).sub(2)
        if(ax is None):
            ax = plt.subplot(projection=wcs)
        ax.imshow(image, origin='lower', vmin=vmin, vmax=vmax)
        ax.grid(color='white', ls='solid')

        # overlay sources from PyBDSF catalog if present
        if (bdsfCat is not None):
            tx = ax.get_transform('world')
            if (bdsfCatFluxLow is None):
                fluxLow = 0.
            else:
                fluxLow = bdsfCatFluxLow
            if (bdsfCatFluxHigh is None):
                fluxHigh = 1000.
            else:
                fluxHigh = bdsfCatFluxHigh
            for s in bdsfCat.sources:
                if (s.peakFlux > fluxLow) and (s.peakFlux < fluxHigh):
                    ax.scatter(s.ra.value, s.dec.value, transform=tx, s=10,
                               edgecolor='white', facecolor='none')
        
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
        # verify the name and get the index of the hdu
        i = self._checkInputName(name)
        image = self.getMap(name)
        self.getWeight()
        if(pos is None):
            pos = (self.headers[i]['CRPIX1'], self.headers[i]['CRPIX2'])
        
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


    def fitGaussian(self, name, pos=None,
                    size=(40, 40),
                    ax=None,
                    image=None,
                    weight=None,
                    wcs=None,
                    plotCutout=True,
                    plotFull=False,
                    plotConvolved=False,
                    returnImage=False,
                    onlyMakeCutout=False,
                    verbose=True,
                    vmin=None, vmax=None):

        # verify the name and get the index of the hdu
        i = self._checkInputName(name)
        image = self.getMap(name)
        self.getWeight()
        if(wcs is None):
            wcs = self.getMapWCS(name)            
        kernel = Gaussian2DKernel(x_stddev=3)
        convolvedImage = convolve(image, kernel)
        if(pos is None):
            pos = np.unravel_index(np.argmax(convolvedImage),
                                   image.shape)
            pos = np.roll(pos, 1)

        cutout = Cutout2D(image, pos, size, mode='partial', fill_value=0.,
                          wcs=wcs)
        weightout = Cutout2D(self.weight, pos, size, mode='partial',
                             fill_value=0., wcs=wcs)
        wcsCutout = cutout.wcs
        Z = cutout.data
        W = weightout.data
        X, Y = np.mgrid[0:size[0], 0:size[1]]
        Xs = np.zeros((X.shape[0], X.shape[1]))
        Ys = np.empty_like(Xs)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                xy = wcsCutout.pixel_to_world(X[i,j], Y[i,j])
                Xs[i,j] = xy.ra.to(u.arcmin).value
                Ys[i,j] = xy.dec.to(u.arcmin).value
                #Xs[i,j] = xy.ra.value
                #Ys[i,j] = xy.dec.value

        if(onlyMakeCutout):
            return None, Z, wcsCutout
        
        # Fit the data using astropy.modeling
        # Dependent on the initial values. If coordinates
        # in degrees no solution is found
        # Then, coordinates are in arcmin
        p_init = models.Gaussian2D(amplitude=Z.max(), x_mean=Xs.mean(),
                                   y_mean=Ys.mean(),
                                   x_stddev=0.01, y_stddev=0.01,
                                   theta=3.*np.pi/2.)#,bounds={'x_stddev':(None,0.5),'y_stddev':(None,0.5)})
        fit_p = fitting.LevMarLSQFitter(calc_uncertainties=True)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',
                                    message='Model is linear in parameters',
                                    category=AstropyUserWarning)
            p = fit_p(p_init, Xs, Ys, Z, weights=W)
            pcov = fit_p.fit_info['param_cov']
            fit = (p, pcov)

        x0, y0 = wcsCutout.world_to_pixel(SkyCoord(ra=p.x_mean.value*u.arcsec,
                                          dec=p.y_mean.value*u.arcsec))        
        centerpos = (x0, y0)
        xy = cutout.to_original_position(centerpos)
        lonlat = wcs.pixel_to_world(xy[0], xy[1])

        if(plotConvolved):
            Z = Cutout2D(convolvedImage, pos, size,
                         mode='partial', fill_value=0.,
                         wcs=wcs).data            
    
        if(plotCutout):
            if(ax is None):
                plt.figure(2)
                plt.clf()
                ax = plt.subplot(projection=wcsCutout)
            else:
                ax = ax
            ax.imshow(Z, origin='lower', vmin=vmin, vmax=vmax)
            ax.axvline(x=y0)
            ax.axhline(y=x0)
            ells = Ellipse(xy=(y0, x0),
                           width=p.x_stddev.value*2.355, height=p.y_stddev.value*2.355,
                           angle=np.rad2deg(p.theta.value)-90.,
                           facecolor='none',
                           edgecolor='red')
            ax.add_artist(ells)
            ells.set_clip_box(ax.bbox)
    
        if(plotFull):
            plt.figure(3)
            plt.clf()
            ax = plt.subplot(projection=wcs)
            if(plotConvolved):
                ax.imshow(convolvedImage)
            else:
                ax.imshow(image, vmin=vmin, vmax=vmax)
            ax.axvline(x=xy[0])
            ax.axhline(y=xy[1])
            ells = Ellipse(xy=xy,
                           width=p.x_stddev.value*2.355,
                           height=p.y_stddev.value*2.355,
                           angle=np.rad2deg(p.theta.value)-90.,
                           facecolor='none',
                           edgecolor='red')
            ax.add_artist(ells)
            ells.set_clip_box(ax.bbox)


        if(returnImage):
            if(plotConvolved):
                return [fit, Z, wcsCutout]
            else:
                return [fit, Z, wcsCutout]
        else:
            return fit


    def _constructKernelInterp(self):
        """Builds a 2-d interpolation of the kernel image.  This is 
        useful for source modeling, addition and subtraction."""
        k = self.getMap('kernel_I')
        r, pos = self.fitImageToGaussian('kernel_I', verbose=False)
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
        # verify the name and get the index of the hdu
        i = self._checkInputName(name)
        if image is None:
            image = self.getMap(name)
        self.getWeight()
        # Deal with the inputs.  If pos is not given, head for center
        # of the map.
        if(sourcePos is None):
            sourcePos = (int(self.headers[i]['CRPIX1']),
                         int(self.headers[i]['CRPIX2']))
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
        
        # create a (100x100) cutout for performing the subtraction
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
        wcs = WCS(self.headers[i]).sub(2)
        
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
        # verify the name and get the index of the hdu
        i = self._checkInputName(name)
        if image is None:
            image = self.getMap(name)
        # Deal with the inputs.  If pos is not given, head for center
        # of the map.
        if(sourcePos is None):
            sourcePos = (int(self.headers[i]['CRPIX1']),
                         int(self.headers[i]['CRPIX2']))
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
    
    
    def photPS(self,cat,method='psf',xy_fixed=True,fitfact=2.5,radius_ap=20.*u.arcsec,inphot=False):
        """Performs Point Source photometry from the signal image and a catalog of positions.
        Inputs:
         - cat [BdsfCat class], the catalog from PyBDSF already read as a BdsfCat class.
         - method [string], optional - default 'psf'. Method used in the photometry estimation. Available pptions:  'psf', 'aperture'.
         - xy_fixed [bool], optional - default True. Fixed center positions fixed or let as free parameters  in the case of psf photometry. 
         - fitfact [float], optional - default 2.5. Factor to apply to the beam size to use as a fitshape parameter in the case of psf photometry.
         - radius_ap [astropy.units.Quantity], optional -  default 20.*u.arcsec. Radius of the circular aperture in the case of aperture photometry.
         - inphot (bool) - if the input fluxes of the sources are known , i.e., if the photometry is done for a SimuInputSources object. False by default. 
        Outputs:
         - result_tab AstropyTab. AstropyTab object which contains an astropy table with the photometry results, i.e., centroids and fluxes estimations and the initial estimates used to start the fitting process.
         - index_weights [array]. Array reporting the elements of the catalog where the weight is larger than the weightCut set at the ToltecSignalFits object (which is 0 by default). 
         
        """

      
        tfipj=self
        ra=cat.ras
        dec=cat.decs
        
        #hdr0=tfipj.headers[0]
        hdr=tfipj.headers[1]
        array_name=tfipj.array
        ## Ideally data could be in MJy/Sr and no need to use the kernel 
        ## and the beam. However, the reduced data has an error 
        ## on the toMJyPerSr conversion and can not be used. 
        ## More recent version do not have this issue and this 
        ## should be change in order to not use the beam
        beam_fwhm=self.beam*gaussian_sigma_to_fwhm#*abs(hdr['CDELT1'])#*3600.
        beam_fwhm = beam_fwhm.to(u.deg)
        fwhm_to_sigma = 1. / (8 * np.log(2))**0.5
        omega_B=np.pi/(4.*np.log(2))*beam_fwhm**2
        flux_fact=self.to_mJyPerBeam/(omega_B.value/abs(hdr['CDELT1'])**2)
        fitshape=int((fitfact*(beam_fwhm.value/abs(hdr['CDELT1'])))/2)*2+1
        #print('toMJyperSr',self.to_MJyPerSr)
        #flux_fact=self.to_MJyPerSr*1.e9*abs(hdr['CDELT1']*np.pi/180.)**2
        
        image=tfipj.getMap('signal_I')*flux_fact/self.maxKer
        noise=(tfipj.getMap('weight_I'))**(-0.5)*flux_fact/self.maxKer
        w = WCS(hdr)
        w=w.celestial
        
        
        sky = SkyCoord(ra, dec, frame=ICRS, unit=(u.deg, u.deg))
        x, y = w.world_to_pixel(sky)
        weights=tfipj.getMap('weight_I')
        weights_xy=np.empty(len(x))
        flux_0=np.empty(len(x))
        for i in range(len(x)):
            
            if (int(round(x[i])) in range(weights.shape[1])) and (int(round(y[i])) in range(weights.shape[0]-1)):
             weights_xy[i]=weights[int(round(y[i])),int(round(x[i]))]
             if weights_xy[i]>0.:
              #print('weights',weights_xy[i])
              #print(image[int(round(y[i]-fitshape*0.5))])
              #print(image[int(round(y[i]-fitshape*0.5)):int(round(y[i]+fitshape*0.5)),int(round(x[i]-fitshape*0.5)):int(round(x[i]+fitshape*0.5))].flatten())
              flux_0[i]=np.max(image[int(round(y[i]-fitshape*0.5)):int(round(y[i]+fitshape*0.5)),int(round(x[i]-fitshape*0.5)):int(round(x[i]+fitshape*0.5))].flatten())/flux_fact
             else:
              flux_0[i] =-99.  
            else: 
              weights_xy[i]=-99. 
              flux_0[i]
        index_weights=np.where(weights_xy>tfipj.weightCut*np.nanmax(weights))[0]    
        x=x[index_weights]
        y=y[index_weights]
        flux_0=flux_0[index_weights]
        
        med_weight=np.median(weights_xy[index_weights])
        if method=='psf':
            
           sigma_psf=beam_fwhm.to(u.arcsec).value/(abs(hdr['CDELT1'])*3600.)/gaussian_sigma_to_fwhm
           
           
           daogroup = DAOGroup(2.0 * sigma_psf * gaussian_sigma_to_fwhm)
           
           mmm_bkg = MMMBackground()
           
           fitter = LevMarLSQFitter()
           
           psf_model = IntegratedGaussianPRF(sigma=sigma_psf)
           
           psf_model.x_0.fixed = xy_fixed
           
           psf_model.y_0.fixed = xy_fixed
           
           
           
           #for 
           
           
           sources=Table()
           #if len(x)>1:
           sources['x_mean']=x
           sources['y_mean']=y
           sources['flux_0']=flux_0
           #else:
            #sources['x_mean']=[x]
            #sources['y_mean']=[y]
              
           pos = Table(names=['x_0', 'y_0','flux_0'], data=[sources['x_mean'],
           
                                                   sources['y_mean'],sources['flux_0']])
           
           #photometry = DAOPhotPSFPhotometry(crit_separation=9, threshold=10.*(med_weight)**(-0.5)*flux_fact, fwhm=9, psf_model=psf_model, fitshape=fitshape)
           #photometry = IterativelySubtractedPSFPhotometry(finder=None,
                                                #group_maker=daogroup,
                                                #bkg_estimator=mmm_bkg,
                                                #psf_model=psf_model,
                                                #fitter=LevMarLSQFitter(),
                                                #niters=3, fitshape=fitshape)
           photometry = BasicPSFPhotometry(group_maker=daogroup,
                                           bkg_estimator=mmm_bkg,
                                           #bkg_estimator=None,
                                           psf_model=psf_model,
                                           fitter=LevMarLSQFitter(),
                                           fitshape=fitshape)
           
           result_tab = photometry(image=image, init_guesses=pos)
           id_sort=result_tab['id'].argsort()
           result_tab=result_tab[id_sort]

           #result_tab_group=photometry.nstar(image=image, star_groups=result_tab)
        elif method=='aperture':
            positions=[(x[i],y[i]) for i in range(len(x))]
            radius_ap=((radius_ap.to(u.deg))/(abs(hdr['CDELT1'])*u.deg)).value
            aperture = CircularAperture(positions, r=radius_ap)
            annulus_aperture = CircularAnnulus(positions, r_in=radius_ap+20, r_out=radius_ap+25)
            result_tab = aperture_photometry(image, aperture, error=noise)
            aperstats = ApertureStats(image, annulus_aperture)
           
            bkg_median = aperstats.median
            aperture_area = aperture.area_overlap(image)
            total_bkg = bkg_median * aperture_area
            phot_bkgsub = result_tab['aperture_sum'] - total_bkg
            result_tab['total_bkg'] = total_bkg
        
            result_tab['aperture_sum_bkgsub'] = phot_bkgsub
            for col in result_tab.colnames:
        
               result_tab[col].info.format = '%.8g' 
        return AstropyTab(result_tab,array=array_name,inphot=inphot,index_weights=index_weights)
###Segmentation of extended images    
    def segmentation(self,npixels=5,snr=3.5,min_flux=None):
      """ 
      Method to detect the different resolved extended regions on an extended 
      object. Photutils is used. See
      
      https://photutils.readthedocs.io/en/stable/segmentation.html
      
      for more information
      
      Inputs.
           
      -npixels (int) The minimum number of connected pixels, each greater than threshold, that an object must have to be deblended. npixels must be a positive integer.
      -snr (float). The desired signal to noise value to be used for the detection threshold. If min_flux is not None, snr argument is ignored. 
      -min_flux (float or 2D ndarray). The data value or pixel-wise data values to be used for the detection threshold. A 2D threshold array must have the same shape as data.
      
      Outputs
      tbls (astropy.table.Table) - astropy table containing the photometry on the segments identified on the ToltecSignalFits.
      segms_deblend (photutils.segmentation.SegmentationImage). Segments  estimated on the ToltecSignalFits.
      """
                
      tfipj=self
      hdr=tfipj.headers[1]

      flux_fact=self.to_MJyPerSr*1.e9*abs(hdr['CDELT1']*np.pi/180.)**2
      image=tfipj.getMap('signal_I')*flux_fact/self.maxKer
      image[np.where(tfipj.getMap('weight_I')<tfipj.weightCut)]=np.nan
    
      bkg_estimator = MedianBackground()
      mask=np.ones((image.shape[0], image.shape[1]), dtype=bool)
      mask[np.where(image>-9.e38)]=False
      nboxes=15
      boxsizex=int(np.ceil(image.shape[0]/nboxes-1))
      boxsizey=int(np.ceil(image.shape[1]/nboxes-1))
      #print(boxsizex,boxsizey)
      #bkg = Background2D(image, (boxsizex, boxsizey), filter_size=(7, 7), bkg_estimator=bkg_estimator,coverage_mask=mask,fill_value=np.nan)
      bkg = Background2D(image, (50, 50), filter_size=(3, 3), bkg_estimator=bkg_estimator)#,coverage_mask=mask,fill_value=np.nan)
      #print('median rms in mJy/px ',bkg.background_rms_median)
      fits.PrimaryHDU(bkg.background,header=hdr).writeto('back.fits',overwrite=True)
      #fits.PrimaryHDU(image,header=hdr).writeto('backsub.fits',overwrite=True)
      image -= bkg.background  # subtract the background   
      fits.PrimaryHDU(image,header=hdr).writeto('backsub.fits',overwrite=True)
      if min_flux==None:
      

       threshold = snr * bkg.background_rms
      else:
       threshold = min_flux*flux_fact/self.maxKer   
      #threshold = detect_threshold(image, nsigma=5.)
      segm = detect_sources(image, threshold, npixels=npixels)
      #print(segm.shape)
      segm_deblend = deblend_sources(image, segm, npixels=npixels,nlevels=32, contrast=0.001)
      #print(segm_deblend.shape)
      ###############################
      #fits.PrimaryHDU(segm_deblend,header=hdr).writeto('segm_deblend.fits',overwrite=True)
      
      #################################
      
      cat = SourceCatalog(image, segm_deblend)#, convolved_data=image)}
      
      
      tbl = cat.to_table()
      self.extPhotTab=tbl
      return tbl,segm_deblend
  
class ToltecFitsList:
    "A List of toltecSignalFits to perform comparison between observations"
    def __init__(self, toltecfits_list):
        self.toltecfits = toltecfits_list
    def segmentation(self,npixels=5,snr=3.5,min_flux=None):
        tbls=[]
        segms_deblend=[]
        for tfi in  self.toltecfits:
            tbli,segm_deblendi=tfi.segmentation(npixels=npixels,snr=snr,min_flux=min_flux)            
            tbls.append(tbli)
            segms_deblend.append(segm_deblendi)
        return  tbls,segms_deblend   
