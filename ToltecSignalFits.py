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
from AstropyTab import AstropyTab
from astropy.coordinates import SkyCoord  
from astropy.coordinates import ICRS
from astropy.table import Table
from photutils.psf import IntegratedGaussianPRF, DAOGroup, BasicPSFPhotometry
from photutils.background import MMMBackground
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import gaussian_sigma_to_fwhm
from photutils.aperture import aperture_photometry, CircularAperture, CircularAnnulus, ApertureStats


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
        r, pos = self.fitImageToGaussian('kernel_I', verbose=False)
        self.beam = r.params['fwhmx'].value*u.arcsec
        self.kerFunc = None

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
    
    
#########################################################    
####Method to perform photometry    
#########################################################3
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
        beam_fwhm=self.beam*abs(hdr['CDELT1'])*3600.
        beam_fwhm = beam_fwhm.to(u.deg)
        fwhm_to_sigma = 1. / (8 * np.log(2))**0.5
        omega_B=np.pi/(4.*np.log(2))*beam_fwhm**2
        flux_fact=self.to_mJyPerBeam/(omega_B.value/abs(hdr['CDELT1'])**2)
        
        
        image=tfipj.getMap('signal_I')*flux_fact
        noise=(tfipj.getMap('weight_I'))**(-0.5)*flux_fact
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
              flux_0[i]=image[int(round(y[i])),int(round(x[i]))]/flux_fact
            else: 
              weights_xy[i]=-99. 
              flux_0[i]=-99.
        index_weights=np.where(weights_xy>tfipj.weightCut*np.nanmax(weights))    
        x=x[index_weights]
        y=y[index_weights]
        flux_0=flux_0[index_weights]
        
        med_weight=np.median(weights_xy[index_weights])
        if method=='psf':
            
           sigma_psf=beam_fwhm.to(u.arcsec).value/(abs(hdr['CDELT1'])*3600.)/gaussian_sigma_to_fwhm
           fitshape=int((fitfact*(beam_fwhm.value/abs(hdr['CDELT1'])))/2)*2+1
           
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
                                           psf_model=psf_model,
                                           fitter=LevMarLSQFitter(),
                                           fitshape=fitshape)
           
           result_tab = photometry(image=image, init_guesses=pos)

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
        return AstropyTab(result_tab,inphot=inphot),index_weights 
