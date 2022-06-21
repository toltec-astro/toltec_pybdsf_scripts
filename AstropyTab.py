from astropy.table import Table
from matplotlib import pyplot as plt

plt.ion()

class AstropyTab:
    """Utility class for reading and working with Astropy tables """
    def __init__(self, astrotab, array='a1100',inphot=False):
        """
        Creating a AstropyTab object will read in the astropy table.
        Inputs:
           catFile (astropy.table.Table) - astropy table
           array (string) - one of 'a1100' (default), 'a1400', or 'a2000'
           inphot (bool) - if the astropy table has the input known sources defined, i.e., if the photometry has been done from a SimuInputSources object. 
           False by default. 
        """
        self.astrotab=astrotab
        self.array = array
        self.inphot=inphot



    def plotInPhot(self):
        """
        Plots the comparison between the input known flux and that 
        estimated from the observation. 
        """
        if self.inphot==False:
            print('This table has not known input fluxes, so comparison between input fluxes and observed fluxes can not be done.')
            
        else:
            plt.figure()
            plt.errorbar(self.astrotab['flux_'+self.array+'_input'],self.astrotab['flux_fit'],yerr=self.astrotab['flux_unc'],fmt='.',color='black',ecolor='grey',elinewidth=0.5)
            plt.plot(self.astrotab['flux_'+self.array+'_input'],self.astrotab['flux_'+self.array+'_input'],color='black')
            plt.title(self.array,fontsize=18)
            plt.xlabel('$F_{\\rm{in}}$[mJy]',fontsize=18)
            plt.ylabel('$F_{\\rm{obs}}$[mJy]',fontsize=18)
            plt.ylim(min(self.astrotab['flux_fit']*0.8),max(self.astrotab['flux_fit']*1.2))

    def plotInPhotGroup(self):
        """
        Plots the comparison between the input known flux and that 
        estimated from the observation, grouped when elements from 
        the input catalog are not spatially resolved.
        """
        if self.inphot==False:
            print('This table has not known input fluxes, so comparison between input fluxes and observed fluxes can not be done.')
            
        else:
            plt.figure()
            plt.errorbar(self.astrotab['flux_'+self.array+'_input_group'],self.astrotab['flux_fit_group'],yerr=self.astrotab['flux_unc_group'],fmt='.',color='black',ecolor='grey',elinewidth=0.5)
            plt.plot(self.astrotab['flux_'+self.array+'_input_group'],self.astrotab['flux_'+self.array+'_input_group'],color='black')
            plt.title(self.array+' Grouped',fontsize=18)
            plt.xlabel('$F_{\\rm{in}}$[mJy]',fontsize=18)
            plt.ylabel('$F_{\\rm{obs}}$[mJy]',fontsize=18)
            plt.ylim(min(self.astrotab['flux_fit_group']*0.8),max(self.astrotab['flux_fit_group']*1.2))
            
            
            
            
            
        
