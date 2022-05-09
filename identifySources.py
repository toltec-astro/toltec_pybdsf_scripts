from ToltecSignalFits import ToltecSignalFits
from astropy.io import fits
from glob import glob
import shutil
import bdsf
import os

# Results directory
# results will be moved here after code completes
resultsDir = 'foo1'
if(not os.path.isdir(resultsDir)):
    os.mkdir(resultsDir)
else:
    p1 = "Delete results directory before running "
    p2 = " or select a new results directory."
    raise OSError(p1+p2)

# Path to the toltec signal file
p1 = "/work/toltec/mapMakerTesting/clusterSims/cluster1/"
p2 = "cluster1_amq25_redu/redu00/"
p3 = "coadded/filtered/"
toltecFitsPath = p1+p2+p3
array = 'a1100'

# Set the weight cut and create the intermediate FITS file
bdsfInputMap = "foo.fits"
rmsMap = bdsfInputMap.replace('.fits', '_rms.fits')
meanMap = bdsfInputMap.replace('.fits', '_mean.fits')
rmsMeanMap = [rmsMap, meanMap]
t1p1 = ToltecSignalFits(toltecFitsPath, array=array)
t1p1.weightCut = 0.1
noise = t1p1.writeImageToFits('signal', bdsfInputMap,
                              JyPerBeam=True,
                              rmsMeanMap=rmsMeanMap,
                              overwrite=True)

# Run the image, rms, and mean maps through PyBDSF to search for
# sources
if(1):
    img = bdsf.process_image(bdsfInputMap, 
                             # rms_map=False, 
                             # mean_map='const', 
                             # rms_value=noise,
                             rmsmean_map_filename=rmsMeanMap,
                             fix_to_beam=True,
                             thresh='fdr', 
                             thresh_pix=5.,
                             output_all=True, 
    )


# Move all the produced files into resultsDir from above
if(1):
    outlist = [bdsfInputMap, rmsMap, meanMap,
               bdsfInputMap.replace('.fits', '.fits.pybdsf.log'),
               bdsfInputMap.replace('.fits', '_pybdsm')]
    for g in outlist:
        shutil.move(g, resultsDir+'/.')
