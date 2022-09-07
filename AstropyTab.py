from astropy.table import Table
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
plt.ion()

class AstropyTab:
    """Utility class for reading and working with Astropy tables """
    def __init__(self, astrotab, array='a1100',inphot=False,index_weights=[np.zeros(0)],fitslist=None):
        """
        Creating a AstropyTab object will read in the astropy table.
        Inputs:
           catFile (astropy.table.Table) - astropy table
           array (string) - one of 'a1100' (default), 'a1400', or 'a2000'
           inphot (bool) - if the astropy table has the input known sources defined, i.e., if the photometry has been done from a SimuInputSources object. 
           False by default. 
        """
        if type(astrotab) != list:
            astrotab=[astrotab]
        self.astrotab=astrotab
        self.array = array
        self.inphot=inphot
        if type(index_weights) == list:
         if len(index_weights)> 0.:
          self.index_weights=index_weights
         else:
          self.index_weights=[np.arange(len(astrotab))]
        else:
         self.index_weights=[index_weights] 
        if type(fitslist)==list: 
         self.fitslist=fitslist
        else:
         self.fitslist=[str(i) for i in range(len(self.astrotab))]   
        #else:
         #self.index_weights=np.arange(len(astrotab))  



    def plotInPhot(self,title=None):
        """
        Plots the comparison between the input known flux and that 
        estimated from the observation. A title label can be added
        """
        if self.inphot==False:
            print('This table has not known input fluxes, so comparison between input fluxes and observed fluxes can not be done.')
            
        else:
          plt.figure()
          if title != None:
           plt.suptitle(title)
          cont=0
          colors=list(mcolors.TABLEAU_COLORS.keys())
          length_colors=len(colors)
          
          for astrotabi in self.astrotab:
            
            index_nonoise=np.where(astrotabi['flux_unc']/astrotabi['flux_fit']<1.)
            plt.errorbar(astrotabi['flux_'+self.array+'_input'][index_nonoise],astrotabi['flux_fit'][index_nonoise],yerr=astrotabi['flux_unc'][index_nonoise],fmt='.',color=colors[np.mod(cont,length_colors)],ecolor='grey',elinewidth=0.5,label=self.fitslist[cont],alpha=0.4)
            

            #plt.ylim(min(astrotabi['flux_'+self.array+'_input'])*0.4,max(astrotabi['flux_'+self.array+'_input'])*2.2)
            #plt.ylim(min(astrotabi['flux_fit'])*0.4,max(astrotabi['flux_fit'])*2.2)
            cont=cont+1
          plt.plot(astrotabi['flux_'+self.array+'_input'],astrotabi['flux_'+self.array+'_input'],color='black')
          plt.title(self.array,fontsize=18)
          plt.legend()
          plt.xlabel('$F_{\\rm{in}}$[mJy]',fontsize=18)
          plt.ylabel('$F_{\\rm{obs}}$[mJy]',fontsize=18)
          plt.yscale('log')
          plt.xscale('log')  
          plt.tight_layout()

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
            plt.ylim(min(self.astrotab['flux_'+self.array+'_input_group'])*0.4,max(self.astrotab['flux_'+self.array+'_input_group'])*2.2)
            plt.yscale('log')
            plt.xscale('log')        
           
    def plotInPhotPos(self,center=(0,0),title=None):
        """
        Plots the ratio between the observed flux and the known input one versus position 
        in the field of view. 
        A title label can be added.
        """
        if self.inphot==False:
            print('This table has not known input fluxes, so comparison between input fluxes and observed fluxes can not be done.')
            
        else:
            cont=0
            colors=list(mcolors.TABLEAU_COLORS.keys())
            length_colors=len(colors)
            fig, axs = plt.subplots(3,figsize=(8,20))
            if title !=None:
             fig.suptitle(title)
            fig.subplots_adjust(hspace=1.0)
            center=(int(np.round(center[0])),int(np.round(center[1])))
            for astrotabi in self.astrotab:
              index_nonoise=np.where(astrotabi['flux_unc']/astrotabi['flux_fit']<1.)
              radius=((astrotabi['x_fit'][index_nonoise]-center[0])**2+(astrotabi['y_fit'][index_nonoise]-center[1])**2)**0.5
              axs[0].plot(radius,astrotabi['flux_fit'][index_nonoise]/astrotabi['flux_'+self.array+'_input'][index_nonoise],'.',color=colors[np.mod(cont,length_colors)],label=self.fitslist[cont],alpha=0.6)
              axs[0].set_title(self.array,fontsize=18)
              axs[0].set_xlabel('Rad [pix]',fontsize=18)
              axs[0].set_ylabel('$\\frac{F_{\\rm{obs}}}{F_{\\rm{in}}}$',fontsize=18)
              
              axs[1].plot(astrotabi['x_fit'][index_nonoise],astrotabi['flux_fit'][index_nonoise]/astrotabi['flux_'+self.array+'_input'][index_nonoise],'.',color=colors[np.mod(cont,length_colors)],label=self.fitslist[cont],alpha=0.6)
              axs[1].set_title(self.array,fontsize=18)
              axs[1].set_xlabel('x [pix]',fontsize=18)
              axs[1].set_ylabel('$\\frac{F_{\\rm{obs}}}{F_{\\rm{in}}}$',fontsize=18)  
              
              axs[2].plot(astrotabi['y_fit'][index_nonoise],astrotabi['flux_fit'][index_nonoise]/astrotabi['flux_'+self.array+'_input'][index_nonoise],'.',color=colors[np.mod(cont,length_colors)],label=self.fitslist[cont],alpha=0.6)
              axs[2].set_title(self.array,fontsize=18)
              axs[2].set_xlabel('y [pix]',fontsize=18)
              axs[2].set_ylabel('$\\frac{F_{\\rm{obs}}}{F_{\\rm{in}}}$',fontsize=18)                
              cont=cont+1
            axs[0].legend()
            axs[0].set_yscale('log')
            axs[0].set_xscale('log') 
            axs[1].set_yscale('log')
            axs[1].set_xscale('log')  
            axs[2].set_yscale('log')
            axs[2].set_xscale('log')              
            

                        
            
        
