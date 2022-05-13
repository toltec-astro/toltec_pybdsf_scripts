from ToltecSignalFits import ToltecSignalFits
from astropy.io import fits
from glob import glob
import shutil
import bdsf
import sys
import os

# Results directory
# results will be moved here after code completes
resultsDir = 'foo1'
if(not os.path.isdir(resultsDir)):
    os.mkdir(resultsDir)
    sys.tracebacklimit = None
else:
    sys.tracebacklimit = 0
    p1 = "Delete results directory before running "
    p2 = " or select a new results directory."
    raise OSError(p1+p2)

# Path to the toltec signal file
p1 = "/old/scratch/toltec_umass_edu/wilson/ItziarCat/"
p2 = "analysis/squareCat_toast_redu/n2c9/redu03/"
p3 = "coadded/filtered/"
toltecFitsPath = p1+p2+p3
array = 'a1100'

# Set the weight cut and create the intermediate FITS file
bdsfInputMap = "foo.fits"
rmsMap = bdsfInputMap.replace('.fits', '_rms.fits')
meanMap = bdsfInputMap.replace('.fits', '_mean.fits')
s2nMap = bdsfInputMap.replace('.fits', '_s2n.fits')
rmsMeanMap = [rmsMap, meanMap]
t1p1 = ToltecSignalFits(toltecFitsPath, array=array)
t1p1.weightCut = 0.5
noise = t1p1.writeImageToFits('signal', bdsfInputMap,
                              JyPerBeam=True,
                              rmsMeanMap=rmsMeanMap,
                              overwrite=True)
_ = t1p1.writeImageToFits('sig2noise', s2nMap, overwrite=True)

# Run the image, rms, and mean maps through PyBDSF to search for
# sources
if(1):
    img = bdsf.process_image(bdsfInputMap, 
                             rms_map=False, 
                             mean_map='const', 
                             # rms_value=noise, 
                             # rmsmean_map_filename=rmsMeanMap,
                             fix_to_beam=True,
                             thresh=None, 
                             thresh_pix=5., 
                             output_all=True,
                             beam=(0.0013889, 0.0013889, 0.0),
                             detection_image=s2nMap,
    )


# Move all the produced files into resultsDir from above
if(1):
    outlist = [bdsfInputMap, rmsMap, meanMap, s2nMap,
               bdsfInputMap.replace('.fits', '.fits.pybdsf.log'),
               bdsfInputMap.replace('.fits', '_pybdsm'),
               s2nMap.replace('.fits', '_pybdsm')]
    for g in outlist:
        shutil.move(g, resultsDir+'/.')
