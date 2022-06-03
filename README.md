### Utilities to work with TolTEC images and PyBDSF

Author: Grant Wilson

### Description

#### ToltecSignalFits.py

This is a continually growing class that works directly with the Citlali output
FITS files.  The member methods are pretty explicit in what they do and the
code is simple enough to see the methods.  But in short, with this you can:
read in images do weight cuts plot images subtract single sources from images
subtract subsets or entire catalogs of sources add sources Of course, all of
these tasks can probably be improved, but I'm pretty happy so far with how they
work.

##### Usage Example: 
```
  from ToltecSignalFits import ToltecSignalFits
  tf1p1 = ToltecSignalFits(<path to citlali output fits files>, array='a1100')
  tf1p1.setWeighCut(0.5)
  tf1p1.plotMap('signal_I', vmin=-0.1, vmax=3.)
```
  
#### BdsfCat.py

This is a helper class for reading in and reformatting PyBDSF catalogs.  This
is much less sophisticated than ToltecSignalFits.py but I find it useful all
the same.  The class pulls data from the ...srl.FITS file in the catalog directory
of the pyBdsf reduction.
 
Usage Example:
```
  from BdsfCat import BdsfCat
  pb = BdsfCat(<path to srl.FITS file>)
```

#### SimuInputSources.py

This is a helper class for reading in and reformatting the simu input source lists
that are used by tolteca simu to populate point sources in simulated maps.

Usage Example:
```
  from SimuInputSources import SumuInputSources
  fluxLimit = 0.2     #in mJy
  sq = SimuInputSources(sourceFile, fluxLimit=fluxLimit)
```

#### CatalogMatch.py

This class manages matching a SimuInputSources catalog with a pyBdsf catalog.
There are several assumptions used in the matching methods and so a user should
read that part of the code carefully.  The output is a set of dictionaries of 
matches, false detections, and unmatched input sources.

Usage Example:
```
  from CatalogMatch import CatalogMatch
  pb = BdsfCat('foo.srl.fits')
  sq = SimInputSources('squareCat.csv')
  cm = CatalogMatch(pb, sq, matchDist=4.*u.arcsec, matchFluxRatioTarget=1.)
  tf = ToltecSignalFits("squarecatReduction/", array='a1100')
  cm.printSummary()
  cm.plotOverlays(tf, vmin=-0.1, vmax=2.0, title='')
```

#### subtractSourcesExample.py

This is a script that shows how to use ToltecSignalFits.py to subtract a
catalog of sources.  Note that it's a script, so you'll have to edit all the
paths to point them someplace useful.

#### identifySources.py

This is a script that uses pyBdsf to identify sources. This will generate a
catalog that the scripts above can read and use.  I don't have pyBdsf installed
on my laptop so I just run this script on Unity.

### How to Contribute

Please share any development you do with these tools by opening a pull request.
