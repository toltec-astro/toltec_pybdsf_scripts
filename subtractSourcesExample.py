from ToltecSignalFits import ToltecSignalFits
from matplotlib import pyplot as plt
import numpy as np
plt.ion()


# simple bright point source test
if(0):
    root = '/Users/wilson/toltecaSims/'
    proj = 'pointSource/'
    pydir = 'PyBDSF/foo1/foo_pybdsm/21Apr2022_18.12.09/catalogues/foo.pybdsm.srl.FITS'
    catFile = root+proj+pydir
    tf = ToltecSignalFits('pointSource/point_reduce/redu00/coadded/filtered/')
    tf.weightCut = 0.1


# cluster1 
if(0):
    root = '/Users/wilson/toltecaSims/'
    proj = 'clusters/cluster1_amq25/'
    pydir = 'foo1/foo_pybdsm/14Apr2022_17.29.59/catalogues/foo.pybdsm.srl.FITS'
    catFile = root+proj+pydir
    tf = ToltecSignalFits(root+proj)
    tf.weightCut = 0.1


# ItziarCat - squareCat
if(1):
    root = '/Users/wilson/toltecaSims/'
    proj = 'ItziarCat/squareCat_toast/'
    pydir = 'foo1/foo_pybdsm/13May2022_14.18.48/catalogues/foo.pybdsm.srl.FITS'
    catFile = root+proj+pydir
    tf = ToltecSignalFits(root+proj+'redu03/')
    tf.weightCut = 0.5

    
if(1):
    image = tf.removeBdsfCatalogFromImage('signal',
                                          catFile=catFile,
                                          fluxLimit=0.0,
                                          fluxCorr=1.0)




