from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u
import numpy as np
import csv
import os
from AstropyTab import AstropyTab

class inputSource:
    def __init__(self, name, ra, dec, f1p1, f1p4, f2p0, coords):
        self.name = name
        self.ra = ra
        self.dec = dec
        self.f1p1 = f1p1
        self.f1p4 = f1p4
        self.f2p0 = f2p0
        self.coords = coords


class SimuInputSources:
    def __init__(self, catFile, fluxLimit=0.0, sortArray='a1100'):
        """catFile must be a standard cvs input file for tolteca simu."""
        # check that the catalog file exists
        if not os.path.exists(catFile):
            raise ValueError("{} does not exist.".format(catFile))
        self.catFile = catFile
        self.fluxLimit = fluxLimit
        self.readCat()
        self.sortByFlux(sortArray)
        self.f1p1 = [s.f1p1 for s in self.sources]
        self.f1p4 = [s.f1p4 for s in self.sources]
        self.f2p0 = [s.f2p0 for s in self.sources]
        self.ras = [s.ra for s in self.sources]
        self.decs = [s.dec for s in self.sources]
        self.coords = SkyCoord([s.coords for s in self.sources])
        

    def readCat(self):
        self.sources = []
        with open(self.catFile) as ff:
            reader = csv.reader(ff, delimiter=" ")
            next(reader)
            for r in reader:
                if(float(r[3]) > self.fluxLimit):
                    s = inputSource(r[0],
                                    float(r[1])*u.deg,
                                    float(r[2])*u.deg,
                                    float(r[3]),
                                    float(r[4]),
                                    float(r[5]),
                                    SkyCoord(float(r[1])*u.deg, float(r[2])*u.deg))
                    self.sources.append(s)

    
    def sortByFlux(self, sortArray):
        sorted = []
        if(sortArray == 'a1100'):
            f = [-s.f1p1 for s in self.sources]
        elif(sortArray == 'a1400'):
            f = [-s.f1p4 for s in self.sources]
        elif(sortArray == 'a2000'):
            f = [-s.f2p0 for s in self.sources]
        else:
            raise ValueError("sort array must be one of 'a1100', 'a1400', or 'a2000'")
        w = np.argsort(f)
        s = [self.sources[i] for i in w]
        self.sources = s


    def trimToWeight(self, weight, wcs):
        swc = []
        for s in self.sources:
            px, py = wcs.world_to_pixel(s.coords)
            pos = (px.min(), py.min())
            if(weight[(round(pos[1]), round(pos[0]))] > 0):
                swc.append(s)
        self.sources = swc
        self.f1p1 = [s.f1p1 for s in self.sources]
        self.f1p4 = [s.f1p4 for s in self.sources]
        self.f2p0 = [s.f2p0 for s in self.sources]
        self.ras = [s.ra for s in self.sources]
        self.decs = [s.dec for s in self.sources]
        self.coords = SkyCoord([s.coords for s in self.sources])

    def inphotPS(self,tis):
        """Performs Point Source photometry from an input catalog of known fluxes (SimuInputSources object) in a ToltecSignalFits object (tfipj).
        Inputs:
         - tfipj [ToltecSignalFits class or list of element from that class], the reduced fits file obtained from citlali already read as a ToltecSignalFits. 
        Outputs:
         - phot_tab. List of AstropyTab. AstropyTab object which contains an astropy table with the psf photometry results, i.e., fluxes estimations, uncertainties, and the input fluxes from the SimuInputSources
        """       
        phottabs_list=[]
        index_weights_list=[]
        labels=[]
        if type(tis) != list:
            tis=[tis]
        for tfipj in  tis:   
          astropytab=tfipj.photPS(self,xy_fixed=False,inphot=True)
          phot_tab=astropytab.astrotab[0]
        
          if tfipj.array=='a1100':
           #print(np.array(self.f1p1)[astropytab.index_weights[0]],astropytab.index_weights[0])
           phot_tab['flux_'+tfipj.array+'_input']=np.array(self.f1p1)[astropytab.index_weights[0]][phot_tab['id']-1]
          elif  tfipj.array=='a1400':
           phot_tab['flux_'+tfipj.array+'_input']=np.array(self.f1p4)[astropytab.index_weights[0]][phot_tab['id']-1]
          elif  tfipj.array=='a2000':
           phot_tab['flux_'+tfipj.array+'_input']=np.array(self.f2p0)[astropytab.index_weights[0]][phot_tab['id']-1]   
           
          phot_tab['flux_fit_group'] = np.empty(len(phot_tab))
          phot_tab['flux_unc_group'] = np.empty(len(phot_tab))
          phot_tab['flux_'+tfipj.array+'_input_group'] = np.empty(len(phot_tab))
          groupsid=list(set(phot_tab['group_id'])) 
          #flux_in=np.empty(len(groupsid))
          #flux_out=np.empty(len(groupsid))
          #flux_out_err=np.empty(len(groupsid))
          for i in groupsid: 
            flux_in=np.sum(phot_tab['flux_'+tfipj.array+'_input'][np.where(phot_tab['group_id']==i)])
            flux_out=np.sum(phot_tab['flux_fit'][np.where(phot_tab['group_id']==i)])
            flux_out_err=np.sum(phot_tab['flux_unc'][np.where(phot_tab['group_id']==i)])
            phot_tab['flux_'+tfipj.array+'_input_group'][np.where(phot_tab['group_id']==i)]=flux_in
            phot_tab['flux_fit_group'][np.where(phot_tab['group_id']==i)]=flux_out
            phot_tab['flux_unc_group'][np.where(phot_tab['group_id']==i)]=flux_out_err 
          phottabs_list.append(phot_tab)
          index_weights_list.append(astropytab.index_weights[0])
          labels.append(tfipj.label)
          #phottabs_list.append(AstropyTab(phot_tab,array=tfipj.array,inphot=True,index_weights=astropytab.index_weights))    
        #return AstropyTab(phot_tab,array=tfipj.array,inphot=True,index_weights=astropytab.index_weights)
        return AstropyTab(phottabs_list,array=tfipj.array,inphot=True,index_weights=index_weights_list,fitslist=labels)
