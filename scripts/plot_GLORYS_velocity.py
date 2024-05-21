#%%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean as cmo
import numpy.ma as ma
from matplotlib import colors as c
import os

#%% Load further utilities
import swm_utilities

#%% Get the data directories 
url_root_output = swm_utilities.load_data_directories()

#%% Get the land mask
land, ds_topo = swm_utilities.get_topograpy()

#%% Get the Glorys velocity field
ds_glorys = swm_utilities.get_glorys_climatology()
uo = ds_glorys.uo.isel(depth=[30,31]).mean(dim='depth').mean(dim='time').values  # 750: 33 # 375: 29 # 225: 26

#%% Get the oxygen data
ds_oxy = swm_utilities.get_oxygen_data()
oxyg = np.squeeze(ds_oxy.o_an.isel(depth=36).values)      # 750: 41 # 375: 31 # 225: 25

#%%
savefig = False
panel_labs = ['a)']
panel_doc = [r'$\overline{u}$ in ms$^{-1}$'+'\n'+r'$\overline{O_2}$ in $\mu$mol kg$^{-1}$'+'\nat '+'~500m']
props = dict(boxstyle='round', facecolor='#FFFFFF', alpha=0.9)
fontsize=18
plt.rcParams['font.size']=fontsize
fig, ax = plt.subplots(1,1,figsize=(8,6),sharex=True,sharey=True)
ax.set_facecolor('#BBBBBB')
c11 = ax.contourf(ds_glorys.longitude.values,ds_glorys.latitude.values,uo,levels=np.linspace(-0.05,0.05,21),cmap='cmo.balance',extend='both')
levs = [30,60,90,120,150,180,210]
c12 = ax.contour(ds_oxy.lon.values,ds_oxy.lat.values,oxyg,levels=levs,colors='k',linewidths=np.linspace(3.5,1.5,len(levs)),zorder=5)
#plt.clabel(c12,inline_spacing=-13,manual=[(-21,8),(-38,10),(-50,16),(-55,20),(-42,35),(0,-5),(5,-10)],rightside_up=True)
plt.clabel(c12,inline_spacing=-22,manual=[(-19,8),(-37,8),(-40,16),(-50,20),(-15,35),(0,-5),(5,-10),(-30,-15),(-25,25)],fmt='{:3.0f} '.format)
ticks = np.linspace(-0.05,0.05,6)
cbar = plt.colorbar(c11,ax=ax,pad=0.03,ticks=ticks)
cbar.ax.tick_params(labelsize=fontsize-4)
# draw the box
ax.plot([-26,-20],[15,15],linewidth=4,color='k')
ax.plot([-26,-20],[9,9],linewidth=4,color='k')
ax.plot([-20,-20],[9,15],linewidth=4,color='k')
ax.plot([-26,-26],[9,15],linewidth=4,color='k')
ax.plot([-26,-20],[15,15],linewidth=3,color='C1')
ax.plot([-26,-20],[9,9],linewidth=3,color='C1')
ax.plot([-20,-20],[9,15],linewidth=3,color='C1')
ax.plot([-26,-26],[9,15],linewidth=3,color='C1')
# draw 23°W
#ax.plot([-23,-23],[-5,14],linewidth=4,color='k')
#ax.plot([-23,-23],[-5,14],linewidth=3,color='w')
axi = ax
idx = 0
axi.pcolormesh(ds_topo.LONGITUDE1,ds_topo.LATITUDE,land,cmap=c.ListedColormap(['#777777']))
axi.text(0.04,0.04,panel_labs[idx],transform=axi.transAxes,va='bottom',ha='left',bbox=props)
if idx == 2 or idx == 3:
    ypos_text = 0.52
else:
    ypos_text = 0.55
axi.text(0.812,ypos_text,panel_doc[idx],transform=axi.transAxes,va='center',ha='center',bbox=props,fontsize=fontsize-1)
for hlineval in [0,20,40]:
    axi.axhline(hlineval,linestyle='--',color='k',alpha=0.1,linewidth=0.75)
for vlineval in [-60,-40,-20,0]:
    axi.axvline(vlineval,linestyle='--',color='k',alpha=0.1,linewidth=0.75)
    axi.set_xticks([-60,-40,-20,0])
    axi.set_xticklabels(['60°W','40°W','20°W','0°'],fontsize=fontsize-3)
    axi.set_yticks([-20,0,20,40])
    axi.set_yticklabels(['20°S','Eq.','20°N','40°N'],fontsize=fontsize-3)
ax.set_xlim([-65,15])
ax.set_ylim([-20,50])
plt.tight_layout()
plt.subplots_adjust(right=0.95,top=0.96)
if savefig == True:
    figname = '../plots/fig_intro_figure_500m'
    plt.savefig('{}.png'.format(figname),dpi=250)
    os.system('convert {}.png {}.pdf'.format(figname,figname))
plt.show()
# %%

# %%
