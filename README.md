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

#### BdsfCat.py

This is a helper class for reading in and reformatting PyBDSF catalogs.  This
is much less sophisticated than ToltecSignalFits.py but I find it useful all
the same.

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
