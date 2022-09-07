from astropy.io import fits
from reproject import reproject_interp
from astropy.convolution import Gaussian2DKernel, convolve,convolve_fft
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.wcs import WCS
from photutils.segmentation import SourceCatalog
import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.psf import  CosineBellWindow, create_matching_kernel
from photutils.psf.matching import resize_psf
import matplotlib as mpl
from astropy.visualization import simple_norm
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources
from photutils.segmentation import SourceCatalog
from photutils.segmentation import deblend_sources
import photutils
import matplotlib.colors as mcolors
mpl.rc('xtick', labelsize=8) 
mpl.rc('ytick', labelsize=8) 
plt.ion()

class InputModelFits:
    
    """Read the input, reproject it, convolve it, and estimate photometry in the segments. 
    """
    def __init__(self, filename=None,array='a1100',kernelFile=None,toltecsignalfits=None,segment=None):
        """Instantiator for a InputModelFits object.
        Inputs: 
          -filename (string) - the path and filename of the input model FITS file.
          -kernelFile (string) - tha path and filename of the kernel which converts from model input fits resolution to the TolTEC observation resolution
          -toltecsignalfits (ToltecFitsList or ToltecSignalFits) - The ToltecSignalFits observatoin associated with the model. A list of toltec fits (ToltecFitsList object) is also 
          segment (photutils.segmentation.SegmentationImage or list of segment objects) - If this model is associated to observations where segmentation was estimated, the segments can be used to estimate authomatically the photometry on the input model. 
        """
        self.filename=filename
        self.hdulist=fits.open(self.filename)
        self.convolved=None
        self.kernelFile=kernelFile
        self.reprojected=None
        self.repro_header=None
        #self.exts=[exti.header['EXTNAME'] for exti in self.hdulist]
        self.data=self.hdulist[array].data
        self.header=self.hdulist[array].header
        self.array=array
        self.segment=segment
        
        self.toltecsignalfits=toltecsignalfits
        
           
        if (self.segment != None) & (self.toltecsignalfits != None):
            if type(self.segment)!=list:
               self.segment=[self.segment] 
            if type(self.toltecsignalfits) != list: 
               self.toltecsignalfits=[self.toltecsignalfits]    
            self.reprojectInputToOutput()
            self.inPhot(self.segment)
      
    def convolveInputToOutput(self):
        "Convolve the observation to the input model resolution "
        hdr=self.header

        data_ker,hdr_ker=fits.open(self.kernelFile)[0].data,fits.open(self.kernelFile)[0].header

        data_ker=resize_psf(data_ker, hdr_ker['CD2_2'], hdr['CDELT2'])


        #print('convolving')
        self.data[np.where(self.data==0.)]=np.nan
        astropy_conv = convolve_fft(self.data,data_ker, fill_value=np.nan, allow_huge=True,normalize_kernel=True) 
        self.convolved=astropy_conv
        fits.PrimaryHDU(astropy_conv,header=self.header).writeto('convolved.fits',overwrite=True)
        return fits.PrimaryHDU(astropy_conv,header=self.header)
        
    def reprojectInputToOutput(self):
        """Reproject a convolved image of the input model to the WCS of 
        the observation, in order to do direct pixel comparison."""
        reprodatas=[]
        hdrs=[]
        hdus=[]
        for toltecsigi in self.toltecsignalfits:
             
         hdr=toltecsigi.headers[1]
         w = WCS(hdr)
         w=w.celestial
         
         reprodata, footprint = reproject_interp(self.convolveInputToOutput(), w,shape_out=(hdr['NAXIS2'],hdr['NAXIS1']))
         reprodatas.append(reprodata)
         hdrs.append(hdr)
         hdus.append(fits.PrimaryHDU(reprodata,header=hdr))
         fits.PrimaryHDU(reprodata,header=hdr).writeto('repro.fits',overwrite=True)
         #reprodata=reprodata#.transpose()
        self.reprojected=reprodatas
        self.repro_header=hdrs
        
        return fits.HDUList(hdus)
    
    def writeReproject(self):
        """Write the convolved and reprojected input fits image with the 
        same name, but with the label '_conv_repro at' the end"""
        flux_fact=1.e9*abs(self.toltecsignalfits.headers[1]['CDELT1']*np.pi/180.)**2
        hdurepro=self.reprojectInputToOutput(self.toltecsignalfits)
        hdurepro.writeto(self.filename.split('.fits')[0]+'_conv_repro.fits',overwrite=True)
        hdu_ratio=fits.PrimaryHDU(hdurepro.data*flux_fact/(self.toltecsignalfits.getMap('signal_I')*flux_fact),header=self.toltecsignalfits.headers[1])
        #hdu_ratio.writeto('ratio.fits',overwrite=True)
        
        
        
    def inPhot(self,segm_deblend):
      """Estimates the photometry on the reprojected and convolved input image on the different input segments´
             Inputs:
          - segm_deblend (photutils.segmentation.SegmentationImage), the segments 
          where perform photometry.
             
             Returns:
             tblin (astropy.table.Table) - astropy table containing the photometry)
       """
      
      tblin=[]
      cont=0
      for  segmi in segm_deblend:
       flux_fact=1.e9*abs(self.repro_header[cont]['CDELT1']*np.pi/180.)**2
       #flux_fact=tfipj.to_mJyPerBeam/(omega_B.value/abs(hdr['CDELT1'])**2)      
       #print(segm_deblend.shape,self.reprojected.shape)
       cati = SourceCatalog(self.reprojected[cont]*flux_fact, segmi)
      
       tblin .append(cati.to_table())
       cont=cont+1
      self.extPhotTab=tblin 
      return tblin
  
    def plotExtPhot(self,title=None):
        """Plots the comparison 1 to 1 of the 
        extended photometry on the segments . 
        A title can be added for labeling purposes."""
        cont=0
        colors=list(mcolors.TABLEAU_COLORS.keys())
        length_colors=len(colors) 
        tfipjs=self.toltecsignalfits
        #if type(tfipjs) != list:
            #tfipjs=[tfipjs]
        plt.figure()
        if title != None:
            plt.title(title)
        for tfipj in tfipjs:
         tblout=tfipj.extPhotTab
        
         
         plt.plot(self.extPhotTab[cont]['segment_flux'],tblout['segment_flux'],'.',color=colors[np.mod(cont,length_colors)],label=tfipj.label,alpha=0.6)
         
         cont=cont+1
        plt.plot(self.extPhotTab[cont-1]['segment_flux'],self.extPhotTab[cont-1]['segment_flux'],color='black') 
        plt.legend() 
        plt.xlabel('$F_{\\rm{in}}$[mJy]')
        plt.ylabel('$F_{\\rm{obs}}$[mJy]')
        plt.yscale('log')
        plt.xscale('log')  
        
        
    def plotModObs(self,segm_deblend,extent=(None,None,None,None),title=None):
     """
      Plots the input model map convolved and reprojected to the 
      observed map, the observed map, and the identified segments. 
      extend can be used to center the plot on the object of interest in image 
      coordinates. Title can be added for labeling purposes. 
      
      Inputs: 
      segm_deblend (photutils.segmentation.SegmentationImage), the segments 
          where photometry is performed.
      extent (tuple-4 elements). (x0,x1,y0,y1), The edges where the map will show on this plot. In image coordinates.     
      title (string).
     """
     if type(segm_deblend) == photutils.segmentation.core.SegmentationImage:
         segm_deblend=[segm_deblend]
     cont=0 
     for tf1p1 in self.toltecsignalfits:
      image=tf1p1.getMap('signal_I')
      hdr=tf1p1.headers[1]
      #x0,x1,y0,y1=260,490,240,490
      if extent[0]!=None:
       x0,x1,y0,y1=extent
      else:
       x0,x1=(0,self.reprojected[cont].shape[0])  
       y0,y1=(0,self.reprojected[cont].shape[1])  
      wcs = WCS(hdr)
      wcs=wcs.celestial
      
      cmap = plt.cm.jet  # define the colormap
      # extract all colors from the .jet map
      cmaplist = [cmap(i) for i in range(cmap.N)]
      # force the first color entry to be grey
      cmaplist[0] = (.5, .5, .5, 1.0)
      
      # create the new map
      cmap = mpl.colors.LinearSegmentedColormap.from_list(
          'Custom cmap', cmaplist, cmap.N)
      
      # define the bins and normalize
      bounds = np.linspace(-1, min(254,max(segm_deblend[cont].labels)), min(254,max(segm_deblend[cont].labels))+1)
      normseg = mpl.colors.BoundaryNorm(bounds, cmap.N)

      
      fig, (ax0,ax1,ax2) = plt.subplots(1, 3,figsize=(13,8),subplot_kw=dict(projection=wcs))
      if title !=None:
       fig.suptitle(title)
      #fig.subplots_adjust(hspace=0.55)
      
      norm = simple_norm(self.reprojected[cont], 'sqrt')
      im0=ax0.imshow(self.reprojected[cont][x0:x1,y0:y1],extent=(x0,x1,y0,y1), origin='lower', cmap='Greys_r',vmin=0,vmax=9)
      ax0.set_title(tf1p1.label+' input model 1100 microns')
      #ax03 = fig.add_axes([.85,0.4,0.03,0.45])
      ax03 = fig.add_axes([.2,0.8,0.4,0.025])
      cbar0 = fig.colorbar(im0,orientation='horizontal',cax=ax03,label='$F_{\\nu}$[MJy/sr]')
      
      #cb0 = mpl.colorbar.ColorbarBase(ax1, norm=norm)
      norm = simple_norm(image, 'sqrt')
      
      im1=ax1.imshow(image[x0:x1,y0:y1],extent=(x0,x1,y0,y1), origin='lower', cmap='Greys_r',vmin=0,vmax=9)
      ax1.set_title(tf1p1.label+' TolTEC simulated 1100 microns')
      #ax13 = fig.add_axes([.91,.124,.04,.754])
      #cbar1 = fig.colorbar(im1,orientation='vertical',cax=ax13)
      ax2.imshow(segm_deblend[cont].data[x0:x1,y0:y1],extent=(x0,x1,y0,y1), origin='lower',cmap=cmap,norm=normseg)
      ax2.set_title(tf1p1.label+' segmented regions')
      #ax3 = fig.add_axes([0.85, 0.05, 0.03, 0.3])
      ax3 = fig.add_axes([.7,0.8,0.2,0.025])
      cb = mpl.colorbar.ColorbarBase(ax3, cmap=cmap, norm=normseg,orientation='horizontal',
      spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i',label='Segment ID') 
      
      cont=cont+1

    def segmentation(self,npixels=5,min_flux=0.):
     """
     Method to detect the different resolved extended regions on an extended 
     object. Photutils is used. See
     
     https://photutils.readthedocs.io/en/stable/segmentation.html
     
     for more information
     
     Inputs.
     -min_flux (float or 2D ndarray). The data value or pixel-wise data values to be used for the detection threshold. A 2D min_flux array must have the same shape as data.      
     -npixels (int) The minimum number of connected pixels, each greater than threshold, that an object must have to be deblended. npixels must be a positive integer.
     
     Outputs
     tbls (astropy.table.Table or list) - astropy table (or list of astropy tables) containing the photometry on the segments identified on the input model map.
     segms_deblend (photutils.segmentation.SegmentationImage or list). Segments (or list of segments) estimated on the input model map.
     """
        
      
     tfipjs=self.toltecsignalfits
     tbls=[]
     segms_deblend=[]
     
     cont=0
     for tfipj in tfipjs:
      hdr=self.repro_header[cont]        
        
      flux_fact=tfipj.to_MJyPerSr*1.e9*abs(hdr['CDELT1']*np.pi/180.)**2
      image_obs=tfipj.getMap('signal_I')*flux_fact/tfipj.maxKer
      #flux_fact=self.to_MJyPerSr*1.e9*abs(hdr['CDELT1']*np.pi/180.)**2
      image=self.reprojected[cont]
      image[np.where(tfipj.getMap('weight_I')<tfipj.weightCut)]=np.nan
      image=image*flux_fact
      
      bkg_estimator = MedianBackground()
      mask=np.ones((image.shape[0], image.shape[1]), dtype=bool)
      mask[np.where(image>-9.e38)]=False
      nboxes=15
      boxsizex=int(np.ceil(image.shape[0]/nboxes-1))
      boxsizey=int(np.ceil(image.shape[1]/nboxes-1))
      #print(boxsizex,boxsizey)
      #bkg = Background2D(image, (boxsizex, boxsizey), filter_size=(7, 7), bkg_estimator=bkg_estimator,coverage_mask=mask,fill_value=np.nan)
      bkg = Background2D(image, (50, 50), filter_size=(3, 3), bkg_estimator=bkg_estimator)
      #print('median rms in mJy/px ',bkg.background_rms_median)
      image_obs -= bkg.background  # subtract the background    
      image_obs[np.where(image_obs<0.*bkg.background_rms_median)]=0.
      threshold = min_flux*flux_fact
      #print(min_flux,min_flux*flux_fact,min_flux*flux_fact*npixels)
      #threshold = detect_threshold(image, nsigma=5.)
      segm = detect_sources(image, threshold, npixels=npixels)
      #print(segm.shape)
      segm_deblend = deblend_sources(image, segm, npixels=npixels,nlevels=32, contrast=0.001)

      cat = SourceCatalog(image_obs, segm_deblend)
      
      cont=cont+1
      tbl = cat.to_table()
      tfipj.extPhotTab=tbl
      tbls.append(tbl)
      segms_deblend.append(segm_deblend)
     self.inPhot(segms_deblend)
     return tbls,segms_deblend      
    
