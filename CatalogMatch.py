from matplotlib import pyplot as plt
import astropy.units as u
import numpy as np
plt.ion()


class CatalogMatch:
    def __init__(self, pyBdsfCat, simInputCat,
                 matchDist=4.*u.arcsec, matchFluxRatioTarget=1.):
        self.matchDist = matchDist
        self.matchFluxRatioTarget = matchFluxRatioTarget
        self.matches = []
        self.falseDetections = []
        self.undetectedInputs = []
        self.pyBdsfCat = pyBdsfCat
        self.simInputCat = simInputCat

        # first use skycoords to construct potential matches based on distance
        idxpb, idxsq, d2d, foo = simInputCat.coords.search_around_sky(
            pyBdsfCat.coords, self.matchDist)

        # trim this list to only include potential matches with reasonable flux ratios
        matches = []
        unique = np.unique(idxpb, return_index=True)[1]
        for i in unique:
            w = np.where(idxpb == idxpb[i])[0]
            if(len(w) == 1):
                ii = i
            elif(len(w) > 1):
                ratios = []
                for j in w:
                    ratios.append(pyBdsfCat.fluxes[idxpb[j]]/simInputCat.f1p1[idxsq[j]])
                ratios = np.array(ratios)
                ii = w[np.argmin(np.abs(ratios-self.matchFluxRatioTarget))]
            y = {
                'ipyBdsfCat': idxpb[ii],
                'isimInputCat': idxsq[ii],
                'fpyBdsfCat': pyBdsfCat.fluxes[idxpb[ii]],
                'fsimInputCat': simInputCat.f1p1[idxsq[ii]],
                'ratio': pyBdsfCat.fluxes[idxpb[ii]]/simInputCat.f1p1[idxsq[ii]],
                'dist': d2d[ii].to(u.arcsec),
                'coords': simInputCat.coords[idxsq[ii]],
            }
            matches.append(y)
        self.matches = matches

        # identify the elements of pb with no match to the input catalog
        # these are false detections
        unmatched = np.where(~np.isin(np.arange(len(pyBdsfCat.sources)), idxpb))[0]
        falseDetections = []
        for i in unmatched:
            y = {
                'i': i,
                'flux': pyBdsfCat.fluxes[i],
                'coords': pyBdsfCat.coords[i],
            }
            falseDetections.append(y)
        self.falseDetections = falseDetections

        # finally, identify any input catalog sources that are undetected
        undet = np.where(~np.isin(np.arange(len(simInputCat.sources)), idxsq))[0]
        undetected = []
        for i in undet:
            y = {
                'i': i,
                'f1p1': simInputCat.f1p1[i],
                'coords': simInputCat.coords[i],
            }
            undetected.append(y)
        self.undetected = undetected
        

    def printSummary(self, ttag=''):
        print("Catalog Comparison"+' '+ttag)
        print("Number of sources: pyBdsfCat: {}, simInputCat: {}".format(
            len(self.pyBdsfCat.fluxes), len(self.simInputCat.sources)))
        print("Number of pyBdsfCat sources matched to simInputCat: {}".format(len(self.matches)))
        print("Number of pyBdsfCat false detections: {}".format(len(self.falseDetections)))
        print("Number of simInputCat sources not found: {}".format(len(self.undetected)))


    def overlayMatches(self, tf, ax=None, vmin=None, vmax=None, title=None):
        if(ax is None):
            ax = tf.plotImage('signal_I', vmin=vmin, vmax=vmax)
        else:
            tf.plotImage('signal_I', ax=ax, vmin=vmin, vmax=vmax)
        tx = ax.get_transform('world')
        for m in self.matches:
            ax.scatter(m['coords'].ra.value, m['coords'].dec.value,
                       transform=tx, s=10,
                       edgecolor='white', facecolor='none')
        if(title is None):
            ax.set_title("PyBDSF sources matched to Sim input sources.")
        else:
            ax.set_title(title)
        return ax


    def overlayFalseDetections(self, tf, ax=None, vmin=None, vmax=None,
                               title=None):
        if(ax is None):
            ax = tf.plotImage('signal_I', vmin=vmin, vmax=vmax)
        else:
            tf.plotImage('signal_I', ax=ax, vmin=vmin, vmax=vmax)
        tx = ax.get_transform('world')
        for m in self.falseDetections:
            ax.scatter(m['coords'].ra.value, m['coords'].dec.value,
                       transform=tx, s=10,
                       edgecolor='red', facecolor='none')
        if(title is None):
            ax.set_title("PyBDSF False Detections.")
        else:
            ax.set_title(title)
        return ax


    def overlayUndetected(self, tf, ax=None, vmin=None, vmax=None, title=None):
        if(ax is None):
            ax = tf.plotImage('signal_I', vmin=vmin, vmax=vmax)
        else:
            tf.plotImage('signal_I', ax=ax, vmin=vmin, vmax=vmax)
        tx = ax.get_transform('world')
        for m in self.undetected:
            ax.scatter(m['coords'].ra.value, m['coords'].dec.value,
                       transform=tx, s=10,
                       edgecolor='green', facecolor='none')
        if(title is None):
            ax.set_title("Sim Input Catalog Undetected Sources.")
        else:
            ax.set_title(title)
        return ax


    def plotOverlays(self, tf, vmin=None, vmax=None, title=None):
        wcs = tf.getMapWCS('signal_I')
        fig = plt.figure(figsize=(14.5, 4.25))
        if(title is not None):
            fig.suptitle(title)
        ax1 = fig.add_subplot(1, 3, 1, projection=wcs)
        ax2 = fig.add_subplot(1, 3, 2, projection=wcs)
        ax3 = fig.add_subplot(1, 3, 3, projection=wcs)
        self.overlayMatches(tf, ax=ax1, vmin=vmin, vmax=vmax, title='BDSF Detections')
        self.overlayFalseDetections(tf, ax=ax2, vmin=vmin, vmax=vmax, title='False Detections')
        self.overlayUndetected(tf, ax=ax3, vmin=vmin, vmax=vmax, title='Undetected')
        
    
    def plotDistVsFluxRatio(self):
        ratios = []
        dists = []
        for m in self.matches:
            ratios.append(m['ratio'])
            dists.append(m['dist'].value)
        plt.figure(1)
        plt.clf()
        plt.xlabel('Distance [arcsec]')
        plt.ylabel('Flux Ratio: pybdsf/catalog')
        plt.plot(dists, ratios, '.k')
        plt.plot([0, np.array(dists).max()], [1, 1], '--k')


    def plotDistHistogram(self):
        dists = []
        for m in self.matches:
            dists.append(m['dist'].value)
        plt.figure(2)
        plt.clf()
        plt.xlabel('Distance [arcsec]')
        plt.ylabel('Histogram')
        plt.hist(dists, bins=50)


    def plotFluxRatioHistogram(self):
        ratios = []
        for m in self.matches:
            ratios.append(m['ratio'])
        plt.figure(3)
        plt.clf()
        plt.xlabel('Flux Ratio: pybdsf/catalog')
        plt.ylabel('Histogram')
        plt.hist(ratios, bins=50)


    def plotFluxVsFluxRatio(self):
        ratios = []
        catFlux = []
        for m in self.matches:
            ratios.append(m['ratio'])
            catFlux.append(m['fsimInputCat'])
        plt.figure(4)
        plt.clf()
        plt.xlabel('Sim Input Catalog Flux [mJy]')
        plt.ylabel('Flux Ratio: pybdsf/catalog')
        plt.plot(catFlux, ratios, '.k')
        plt.plot([0., np.array(catFlux).max()], [1., 1.], '--k')
