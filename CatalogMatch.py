from statsmodels.stats.proportion import proportion_confint
from matplotlib import pyplot as plt, ticker as mticker
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
            self.pyBdsfCat.coords, self.matchDist)

        self.matches = self.constructMatches(idxpb, idxsq, d2d)
        self.falseDetections = self.constructFalseDetections(idxpb)
        self.undetected = self.constructUndetected()


    def constructMatches(self, idxpb, idxsq, d2d):
        # trim this list to only include potential matches with reasonable flux ratios
        siFlux = self.getCorrectSimInputFlux()
        matches = []
        unique = np.unique(idxpb, return_index=True)[1]
        for i in unique:
            w = np.where(idxpb == idxpb[i])[0]
            if(len(w) == 1):
                ii = i
            elif(len(w) > 1):
                ratios = []
                for j in w:
                    ratios.append(self.pyBdsfCat.fluxes[idxpb[j]]/siFlux[idxsq[j]])
                ratios = np.array(ratios)
                ii = w[np.argmin(np.abs(ratios-self.matchFluxRatioTarget))]
            y = {
                'pyBdsfCat': {
                    'indx': idxpb[ii],
                    'flux': self.pyBdsfCat.fluxes[idxpb[ii]],
                    's2n': self.pyBdsfCat.s2ns[idxpb[ii]],
                    'array': self.pyBdsfCat.array,
                    'coords': self.pyBdsfCat.coords[idxpb[ii]],
                    },
                'simInputCat': {
                    'indx': idxsq[ii],
                    'flux': siFlux[idxsq[ii]],
                    'f1p1': self.simInputCat.f1p1[idxsq[ii]],
                    'f1p4': self.simInputCat.f1p4[idxsq[ii]],
                    'f2p0': self.simInputCat.f2p0[idxsq[ii]],
                    'coords': self.simInputCat.coords[idxsq[ii]],
                    'otherPotentials': idxsq[w],
                    },
                'ratio': self.pyBdsfCat.fluxes[idxpb[ii]]/siFlux[idxsq[ii]],
                'dist': d2d[ii].to(u.arcsec),
            }            
            matches.append(y)
        return  matches


    def constructFalseDetections(self, idxpb):
        # identify the elements of pb with no match to the input catalog
        # these are false detections
        unmatched = np.where(~np.isin(np.arange(len(self.pyBdsfCat.sources)), idxpb))[0]
        falseDetections = []
        for i in unmatched:
            y = {
                'pyBdsfCat': {
                    'indx': i,
                    'flux': self.pyBdsfCat.fluxes[i],
                    's2n': self.pyBdsfCat.s2ns[i],
                    'array': self.pyBdsfCat.array,
                    'coords': self.pyBdsfCat.coords[i],
                    },
                }
            falseDetections.append(y)
        return falseDetections


    def constructUndetected(self):
        # finally, identify any input catalog sources that are umatched to pyBdsfCat sources
        siFlux = self.getCorrectSimInputFlux()
        idMatched = []
        for m in self.matches:
            idMatched.append(m['simInputCat']['indx'])
        idMatched = np.array(idMatched)
        undet = np.where(~np.isin(np.arange(len(self.simInputCat.sources)), idMatched))[0]
        undetected = []
        for i in undet:
            y = {
                'pyBdsfCat': {
                    'array': self.pyBdsfCat.array,
                    },
                'simInputCat': {
                    'indx': i,
                    'flux': siFlux[i],
                    'f1p1': self.simInputCat.f1p1[i],
                    'f1p4': self.simInputCat.f1p4[i],
                    'f2p0': self.simInputCat.f2p0[i],
                    'coords': self.simInputCat.coords[i],
                    },
            }
            undetected.append(y)
        return undetected
        

    def getCorrectSimInputFlux(self):
        if(self.pyBdsfCat.array == 'a1100'):
            siFlux = self.simInputCat.f1p1
        elif(self.pyBdsfCat.array == 'a1400'):
            siFlux = self.simInputCat.f1p4
        else:
            siFlux = self.simInputCat.f2p0
        return siFlux
            

    def printSummary(self, ttag=''):
        print("Catalog Comparison"+' '+ttag)
        print("Number of sources: pyBdsfCat: {}, simInputCat: {}".format(
            len(self.pyBdsfCat.sources), len(self.simInputCat.sources)))
        print("Number of pyBdsfCat sources matched to simInputCat: {}".format(len(self.matches)))
        print("Number of pyBdsfCat false detections: {}".format(len(self.falseDetections)))
        print("Number of simInputCat sources not found: {}".format(len(self.undetected)))
        return


    def plotCompleteness(self, bins=None):
        if(bins is None):
            bins = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                             0.8, 0.9, 1.0, 2.0, 3.0, 5.0, 7.0, 10.])
        siFlux = self.getCorrectSimInputFlux()
        hsi, bsi = np.histogram(siFlux, bins)

        matchedSiFlux = []
        for m in self.matches:
            matchedSiFlux.append(m['simInputCat']['flux'])
        hmsi, bmsi = np.histogram(matchedSiFlux, bins)

        frac = []
        bc = []
        for i in range(len(bins)-1):
            bc.append((bins[i+1]+bins[i])/2.)
            if(hsi[i] > 0):
                frac.append((hmsi[i]/hsi[i])*100.)
            else:
                frac.append(0.)
        bc = np.array(bc)

        # use gaussian approximation of binomial distribution for error
        errLow = []
        errHigh = []
        for h, hm in zip(hsi, hmsi):
            if(h > 0):
                low, high = proportion_confint(hm, h, alpha=0.32, method='normal')
                errLow.append(low)
                errHigh.append(high)
            else:
                errLow.append(0.)
                errHigh.append(0.)
        frac=np.array(frac)
        errLow = np.array(errLow)*100.
        errHigh = np.array(errHigh)*100.
        comp_error = [frac-errLow, errHigh-frac]
        
        plt.clf()
        plt.xlabel('Sim Input Flux [mJy]')
        plt.ylabel('Fraction of Matched Sources %')
        plt.title('Completeness of Sim Input Source Detections')
        plt.errorbar(bc, frac, yerr=comp_error, fmt='o', color='blue')
        plt.xscale('log')
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

        return 

        

    def overlayMatches(self, tf, ax=None, vmin=None, vmax=None, title=None):
        if(ax is None):
            ax = tf.plotImage('signal_I', vmin=vmin, vmax=vmax)
        else:
            tf.plotImage('signal_I', ax=ax, vmin=vmin, vmax=vmax)
        tx = ax.get_transform('world')
        for m in self.matches:
            ax.scatter(m['simInputCat']['coords'].ra.value,
                       m['simInputCat']['coords'].dec.value,
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
            ax.scatter(m['pyBdsfCat']['coords'].ra.value,
                       m['pyBdsfCat']['coords'].dec.value,
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
            ax.scatter(m['simInputCat']['coords'].ra.value,
                       m['simInputCat']['coords'].dec.value,
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
            catFlux.append(m['simInputCat'][flux])
        plt.figure(4)
        plt.clf()
        plt.xlabel('Sim Input Catalog Flux [mJy]')
        plt.ylabel('Flux Ratio: pybdsf/catalog')
        plt.plot(catFlux, ratios, '.k')
        plt.plot([0., np.array(catFlux).max()], [1., 1.], '--k')