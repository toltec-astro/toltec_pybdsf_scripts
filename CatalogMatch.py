from statsmodels.stats.proportion import proportion_confint
from matplotlib import pyplot as plt, ticker as mticker
import astropy.units as u
import astropy
import numpy as np
from uncertainties import ufloat
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

    def plotMatch(self,tabphot):
        #tabphot=tabphot
        #print(tabphot.astrotab)
        #print(tabphot.index_weights)
        fluxPSF = []
        fluxPSF_err = []
        catFlux = []
        pybdsfFlux = []
        pybdsfFlux_err = []
        fluxdict={'a1100':'f1p1','a1400':'f1p4','a2000':'f2p0'}
        #self.matches=self.matches[tabphot.index_weights]
        #cont_match=0
        for m in self.matches:
         catFlux.append(m['simInputCat'][fluxdict[tabphot.array]])
         pybdsfFlux.append(m['pyBdsfCat']['flux'])   
         pybdsfFlux_err.append(m['pyBdsfCat']['flux']/m['pyBdsfCat']['s2n'])
         index_tab=np.where( tabphot.index_weights[0]==m['pyBdsfCat']['indx'])[0]
         #print(index_tab)
         if len(index_tab)==1:
            
            fluxPSF.append(tabphot.astrotab[0]['flux_fit'][index_tab][0])
            fluxPSF_err.append(tabphot.astrotab[0]['flux_unc'][index_tab][0])

             
         else:    
            fluxPSF.append(np.nan)  
            fluxPSF_err.append(0.) 
        mean_PSF=np.nanmean(np.array(fluxPSF,dtype=float)/np.array(catFlux,dtype=float))
        std_PSF=np.nanstd(np.array(fluxPSF,dtype=float)/np.array(catFlux,dtype=float))  
        mean_P_BD=np.nanmean(np.array(fluxPSF,dtype=float)/pybdsfFlux)
        std_P_BD=np.nanstd(np.array(fluxPSF,dtype=float)/pybdsfFlux)
        mean_PyBDSF=np.nanmean(pybdsfFlux/np.array(catFlux,dtype=float))
        std_PyBDSF=np.nanstd(pybdsfFlux/np.array(catFlux,dtype=float))  
        
        #print(len(fluxPSF),len(catFlux),len(self.matches))   
        fig, axs = plt.subplots(3,figsize=(8,20))
        #fig.subplots_adjust(wspace=3.0)
        fig.subplots_adjust(hspace=1.0)
        
        axs[0].errorbar(np.array(catFlux,dtype=float),np.array(fluxPSF,dtype=float),yerr=np.array(fluxPSF_err,dtype=float),fmt='.',color='black',ecolor='grey',elinewidth=0.5)
        #axs[0].plot(catFlux,fluxPSF,'.')
        axs[0].annotate('$\\frac{F_{\\rm{PSF}}}{F_{\\rm{in}}}_{\\rm{mean}}=$'+'{:.2u}'.format(ufloat(mean_PSF,std_PSF)), xy=(0.15, 0.85), xycoords='figure fraction',
            size=11, ha='left', va='top',
            bbox=dict(boxstyle='round', fc='w'))
        #axs[0].text(0.1,0.85,'$\\frac{F_{\\rm{PSF}}}{F_{\\rm{in}}}_{\\rm{mean}}=$'+'{:.1u}'.format(ufloat(mean_PSF,std_PSF)), horizontalalignment='center',
        #verticalalignment='center', transform=axs[0].transAxes, bbox=dict(facecolor='black', alpha=0.5))
        axs[0].plot(catFlux,catFlux,color='black')
        axs[0].set_title(tabphot.array+ ' Matched' ,fontsize=18)
        axs[0].set_xlabel('$F_{\\rm{in}}$[mJy]',fontsize=18)
        axs[0].set_ylabel('$F_{\\rm{PSF}}$[mJy]',fontsize=18)
        axs[0].set_ylim(min(np.array(fluxPSF)*0.7),max(np.array(fluxPSF)*1.3))  
        axs[1].annotate('$\\frac{F_{\\rm{PSF}}}{F_{\\rm{pyBDSF}}}_{\\rm{mean}}=$'+'{:.2u}'.format(ufloat(mean_P_BD,std_P_BD)), xy=(0.15, 0.55), xycoords='figure fraction',
            size=11, ha='left', va='top',
            bbox=dict(boxstyle='round', fc='w'))
        #axs[1].text(0.1,0.85,'$\\frac{F_{\\rm{PSF}}}{F_{\\rm{pyBDSF}}}_{\\rm{mean}}=$'+'{:.1u}'.format(ufloat(mean_P_BD,std_P_BD)), horizontalalignment='center',
        #verticalalignment='center', transform=axs[1].transAxes, bbox=dict(facecolor='black', alpha=0.5))
        axs[1].errorbar(np.array(pybdsfFlux,dtype=float),np.array(fluxPSF,dtype=float),yerr=np.array(fluxPSF_err,dtype=float),xerr=np.array(pybdsfFlux_err,dtype=float),fmt='.',color='black',ecolor='grey',elinewidth=0.5)
        axs[1].plot(pybdsfFlux,pybdsfFlux,color='black')
        axs[1].set_title(tabphot.array+ ' Matched' ,fontsize=18)
        axs[1].set_xlabel('$F_{\\rm{pyBDSF}}$[mJy]',fontsize=18)
        axs[1].set_ylabel('$F_{\\rm{PSF}}$[mJy]',fontsize=18)       
        axs[1].set_ylim(min(np.array(fluxPSF)*0.8),max(np.array(fluxPSF)*1.2))   
        
        axs[2].annotate('$\\frac{F_{\\rm{pyBDSF}}}{F_{\\rm{in}}}_{\\rm{mean}}=$'+'{:.2u}'.format(ufloat(mean_PyBDSF,std_PyBDSF)), xy=(0.15, 0.24), xycoords='figure fraction',
            size=11, ha='left', va='top',
            bbox=dict(boxstyle='round', fc='w'))
        #axs[2].text(0.1,0.85,'$\\frac{F_{\\rm{pyBDSF}}}{F_{\\rm{in}}}_{\\rm{mean}}=$'+'{:.1u}'.format(ufloat(mean_PyBDSF,std_PyBDSF)), horizontalalignment='center',
        #verticalalignment='center', transform=axs[2].transAxes, bbox=dict(facecolor='black', alpha=0.5))
        axs[2].errorbar(catFlux,pybdsfFlux,yerr=np.array(pybdsfFlux_err,dtype=float),fmt='.',color='black',ecolor='grey',elinewidth=0.5)
        axs[2].plot(catFlux,catFlux,color='black')
        axs[2].set_title(tabphot.array+ ' Matched' ,fontsize=18)
        axs[2].set_ylabel('$F_{\\rm{pyBDSF}}$[mJy]',fontsize=18)
        axs[2].set_xlabel('$F_{\\rm{in}}$[mJy]',fontsize=18)       
        axs[2].set_ylim(min(np.array(catFlux)*0.8),max(np.array(catFlux)*1.2))           

    def plotPhotPos(self,tabphot,center=(0,0)):
        #tabphot=tabphot
        fluxPSF = []
        #fluxPSF_err = []
        catFlux = []
        radius= []
        xpos=[]
        ypos=[]
        #pybdsfFlux = []
        #pybdsfFlux_err = []
        fluxdict={'a1100':'f1p1','a1400':'f1p4','a2000':'f2p0'}

        for m in self.matches:
         catFlux.append(m['simInputCat'][fluxdict[tabphot.array]])
         #pybdsfFlux.append(m['pyBdsfCat']['flux'])   
         #pybdsfFlux_err.append(m['pyBdsfCat']['flux']/m['pyBdsfCat']['s2n'])
         index_tab=np.where( tabphot.index_weights[0]==m['pyBdsfCat']['indx'])[0]
         #print(index_tab)
         if len(index_tab)==1:
            if type(tabphot.astrotab[0]['flux_fit'][index_tab])==astropy.table.column.Column:
              fluxPSF.append(tabphot.astrotab[0]['flux_fit'][index_tab][0])
              radius.append(((tabphot.astrotab[0]['x_fit'][index_tab][0]-center[0])**2+(tabphot.astrotab[0]['y_fit'][index_tab][0]-center[1])**2)**0.5)
              xpos.append(tabphot.astrotab[0]['x_fit'][index_tab][0])
              ypos.append(tabphot.astrotab[0]['y_fit'][index_tab][0])
            else:
              fluxPSF.append(tabphot.astrotab[0]['flux_fit'][index_tab])
              radius.append(((tabphot.astrotab[0]['x_fit'][index_tab]-center[0])**2+(tabphot.astrotab[0]['y_fit'][index_tab]-center[1])**2)**0.5)
              xpos.append(tabphot.astrotab[0]['x_fit'][index_tab])
              ypos.append(tabphot.astrotab[0]['y_fit'][index_tab])                
            #fluxPSF_err.append(tabphot.astrotab[0]['flux_unc'][index_tab])

             
         else:    
           fluxPSF.append(np.nan)
           radius.append(np.nan)
           xpos.append(np.nan)
           ypos.append(np.nan)
        fig, axs = plt.subplots(3,figsize=(8,20))
        fig.subplots_adjust(hspace=1.0)
        
        axs[0].plot(np.array(radius,dtype=float),np.array(fluxPSF,dtype=float)/np.array(catFlux,dtype=float),'.',color='black')
        axs[0].set_title(tabphot.array+ ' Matched' ,fontsize=18)
        axs[0].set_xlabel('Radius [pix]',fontsize=18)
        axs[0].set_ylabel('$\\frac{F_{\\rm{obs}}}{F_{\\rm{in}}}$',fontsize=18)     
        axs[1].plot(np.array(xpos,dtype=float),np.array(fluxPSF,dtype=float)/np.array(catFlux,dtype=float),'.',color='black')
        axs[1].set_title(tabphot.array+ ' Matched' ,fontsize=18)
        axs[1].set_xlabel('x [pix]',fontsize=18)
        axs[1].set_ylabel('$\\frac{F_{\\rm{obs}}}{F_{\\rm{in}}}$',fontsize=18) 
        
        axs[2].plot(np.array(ypos,dtype=float),np.array(fluxPSF,dtype=float)/np.array(catFlux,dtype=float),'.',color='black')
        axs[2].set_title(tabphot.array+ ' Matched' ,fontsize=18)
        axs[2].set_xlabel('y [pix]',fontsize=18)
        axs[2].set_ylabel('$\\frac{F_{\\rm{obs}}}{F_{\\rm{in}}}$',fontsize=18)          
 


