"""
author: Eike E. Koehn
date: Jan 22, 2024
description: generates figure for means in spun up model (over last 80 years)
"""

#%% Load the packages
print('Load packages.')
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean as cmo
from matplotlib import colors as c
#% Load further utilities
import swm_utilities

#%% Get the data directories 
print('Get data directories.')
url_root_output = swm_utilities.load_data_directories()

#%% Get the land mask
print('Get land mask for plotting later on.')
land, ds_topo = swm_utilities.get_topograpy()

#%% Open the datasets
print('Open the required datasets.')
datasets_to_load = dict()
#
ds_eta = xr.open_dataset('{}eta.nc'.format(url_root_output),decode_times=False)
datasets_to_load['eta'] = dict() 
datasets_to_load['eta']['ds'] = ds_eta
datasets_to_load['eta']['varname'] = 'eta'
#
ds_T1_CH = xr.open_dataset('{}T1_CH.nc'.format(url_root_output),decode_times=False)
datasets_to_load['CH'] = dict() 
datasets_to_load['CH']['ds'] = ds_T1_CH
datasets_to_load['CH']['varname'] = 'T1_CH'
#
ds_mu = xr.open_dataset('{}mu.nc'.format(url_root_output),decode_times=False)
datasets_to_load['mu'] = dict() 
datasets_to_load['mu']['ds'] = ds_mu
datasets_to_load['mu']['varname'] = 'SWM_MU'
#
ds_mv = xr.open_dataset('{}mv.nc'.format(url_root_output),decode_times=False)
datasets_to_load['mv'] = dict() 
datasets_to_load['mv']['ds'] = ds_mv
datasets_to_load['mv']['varname'] = 'SWM_MV'
#
ds_du = xr.open_dataset('{}du.nc'.format(url_root_output),decode_times=False)
datasets_to_load['du'] = dict() 
datasets_to_load['du']['ds'] = ds_du
datasets_to_load['du']['varname'] = 'SWM_DU'
#
ds_dv = xr.open_dataset('{}dv.nc'.format(url_root_output),decode_times=False)
datasets_to_load['dv'] = dict() 
datasets_to_load['dv']['ds'] = ds_dv
datasets_to_load['dv']['varname'] = 'SWM_DV'

#%% Calculate the means over the last 80 years of the model output
print('Calculate the means over the last 80 year of the model output.')

number_of_years = 80
time_minidx = -number_of_years*12 # number of months (negative to select last X number of months using .slice functionality)
#
means_dict = dict()
for var2load in datasets_to_load.keys():
    print(var2load)
    # get the right variable names etc.
    means_dict[var2load] = dict()
    ds_dum       = datasets_to_load[var2load]['ds']
    varname_dum  = datasets_to_load[var2load]['varname']
    # select subset
    selected_t   = ds_dum.isel(TIME=slice(time_minidx,None))
    # calculate the mean and load the mean into memory
    means_dict[var2load] = selected_t.variables[varname_dum].mean(dim='TIME').load()
    print('Calculation for mean of {} done'.format(var2load))

#%% Calculate thickness-weighted averages
print('Calculate thickness weighted averages.')
h0 = 500. # undisturbed layer depth
means_dict['thickness'] = means_dict['eta'] + h0 # convert interface displacement to active layer thickness
Chat = means_dict['CH']/(means_dict['thickness'])
uhat = means_dict['mu']/means_dict['du']
vhat = means_dict['mv']/means_dict['dv']

#%% Generate the plot
#####################
 
print('Set plot properties.')
savefig = False
panel_labs = ['a)','b)','c)','d)']
panel_doc = [r'$\overline{h}$'+'\n'+'in m',r'$\hat{C}$',r'$\hat{u}$'+'\n'+'in ms$^{-1}$',r'$\hat{v}$'+'\n'+'in ms$^{-1}$']
props = dict(boxstyle='round', facecolor='#EEEEEE', alpha=0.9)
fontsize=20
plt.rcParams['font.size']=fontsize
#

print('Generate the plot.')
fig, ax = plt.subplots(2,2,figsize=(13.5,10),sharex=True,sharey=True)
#ax[0,0]
c00_min = 450; c00_max = 510
c00 = ax[0,0].contourf(ds_eta.LONGITUDE,ds_eta.LATITUDE,means_dict['thickness'],levels=np.linspace(c00_min,c00_max,21),cmap='cmo.delta',extend='both')
plt.colorbar(c00,ax=ax[0,0],pad=0.03,ticks=np.linspace(c00_min,c00_max,5))
# ax[0,1]
c01_min = 0; c01_max = 1
c01 = ax[0,1].contourf(ds_T1_CH.LONGITUDE,ds_T1_CH.LATITUDE,Chat,levels=np.linspace(c01_min,c01_max,21),cmap='cmo.haline',extend='neither')
plt.colorbar(c01,ax=ax[0,1],pad=0.03,ticks=np.linspace(c01_min,c01_max,6))
# ax[1,0]
c10_min = -0.05; c10_max = 0.05
c10 = ax[1,0].contourf(ds_du.LONGITUDE,ds_du.LATITUDE,uhat,levels=np.linspace(c10_min,c10_max,21),cmap='cmo.balance',extend='both')
plt.colorbar(c10,ax=ax[1,0],pad=0.03,ticks=np.linspace(c10_min,c10_max,6))
# ax[1,1]
c11_min = -0.05; c11_max = 0.05
c11 = ax[1,1].contourf(ds_dv.LONGITUDE,ds_dv.LATITUDE,vhat,levels=np.linspace(c11_min,c11_max,21),cmap='cmo.balance',extend='both')
plt.colorbar(c11,ax=ax[1,1],pad=0.03,ticks=np.linspace(c11_min,c11_max,6))
# beautify
plt.tight_layout()
plt.subplots_adjust(right=0.95,left=0.07,top=0.96)
for idx,axi in enumerate(ax.flatten()):
    axi.pcolormesh(ds_topo.LONGITUDE1,ds_topo.LATITUDE,land,cmap=c.ListedColormap(['#777777']))
    axi.text(0.04,0.04,panel_labs[idx],transform=axi.transAxes,va='bottom',ha='left',bbox=props)
    if idx == 2 or idx == 3:
        ypos_text = 0.54
    else:
        ypos_text = 0.57
    axi.text(0.82,ypos_text,panel_doc[idx],transform=axi.transAxes,va='center',ha='center',bbox=props,fontsize=fontsize+2)
    for hlineval in [0,20,40]:
        axi.axhline(hlineval,linestyle='--',color='k',alpha=0.1,linewidth=0.75)
    for vlineval in [-60,-40,-20,0]:
        axi.axvline(vlineval,linestyle='--',color='k',alpha=0.1,linewidth=0.75)
for axi in ax[-1,:]:
    axi.set_xticks([-60,-40,-20,0])
    axi.set_xticklabels(['60°W','40°W','20°W','0°'],fontsize=fontsize-3)
for axi in ax[:,0]:
    axi.set_yticks([-20,0,20,40])
    axi.set_yticklabels(['20°S','Eq.','20°N','40°N'],fontsize=fontsize-3)
if savefig == True:
    figname = '../plots/fig_means'
    plt.savefig('{}.png'.format(figname),dpi=250)
    os.system('convert {}.png {}.pdf'.format(figname,figname))
plt.show()
# %%
