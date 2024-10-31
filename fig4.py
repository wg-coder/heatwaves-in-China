#%%
import numpy as np
from netCDF4 import Dataset
import xarray as xr
from datetime import datetime
import pandas as pd
import pickle
import pymannkendall as mk
import glob
import os
from sklearn.linear_model import LinearRegression


# plot libaray
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from cartopy.io.shapereader import Reader
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import cmaps
# import seaborn as sns
import matplotlib.transforms as mtransforms
from matplotlib.ticker import ScalarFormatter



# ssp585 hw exposure change
fig = plt.figure(figsize=(17,22))
grid = plt.GridSpec(18,18, wspace=0.12, hspace = 0.1)
proj_china=ccrs.LambertConformal(central_longitude=107.5,central_latitude=30.0,standard_parallels=(20,50))

# set sub figure
ax1 = fig.add_subplot(grid[0:5,0:8],projection = proj_china)
ax2 = fig.add_subplot(grid[6:11,0:8],projection = proj_china)
ax3 = fig.add_subplot(grid[12:17,0:8],projection = proj_china)
ax4 = fig.add_subplot(grid[0:5,9:17],projection = proj_china)
ax5 = fig.add_subplot(grid[6:11,9:17],projection = proj_china)
ax6 = fig.add_subplot(grid[12:17,9:17],projection = proj_china)

ax1_si = fig.add_axes([0.377, 0.7133, 0.092, 0.06], frameon=True,projection=proj_china)
ax2_si = fig.add_axes([0.377, 0.4553, 0.092, 0.06], frameon=True,projection=proj_china)
ax3_si = fig.add_axes([0.377, 0.1973, 0.092, 0.06], frameon=True,projection=proj_china)
ax4_si = fig.add_axes([0.7667, 0.7133, 0.092, 0.06], frameon=True,projection=proj_china)
ax5_si = fig.add_axes([0.7667, 0.4553, 0.092, 0.06], frameon=True,projection=proj_china)
ax6_si = fig.add_axes([0.7667, 0.1973, 0.092, 0.06], frameon=True,projection=proj_china)

ax1.cla()
(ds_ssp585_0std_gwl15_exposure_change['exposure']).plot(
    ax=ax1,levels=np.arange(-8,8.1,0.5),cmap='BlueYellowRed',transform = ccrs.PlateCarree(),extend='both',
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Exposure (days)')
)
add_map_feature(ax1,cf_china, cf_9line)
(ds_ssp585_0std_gwl15_exposure_change['exposure']).plot(
    ax=ax1_si,levels=np.arange(-8,8.1,0.5),cmap='BlueYellowRed',transform = ccrs.PlateCarree(),extend='both',
    add_colorbar=False
)
add_map_nanhai(ax1_si, cf_china, cf_9line)
fig.show()

ax2.cla()
(ds_ssp585_0std_gwl2_exposure_change['exposure']).plot(
    ax=ax2,levels=np.arange(-8,8.1,0.5),cmap='BlueYellowRed',transform = ccrs.PlateCarree(),extend='both',
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Exposure (days)')
)
add_map_feature(ax2,cf_china, cf_9line)
(ds_ssp585_0std_gwl2_exposure_change['exposure']).plot(
    ax=ax2_si,levels=np.arange(-8,8.1,0.5),cmap='BlueYellowRed',transform = ccrs.PlateCarree(),extend='both',
    add_colorbar=False)
add_map_nanhai(ax2_si, cf_china, cf_9line)

ax3.cla()
(ds_ssp585_0std_gwl05_exposure_change['exposure']).plot(
    ax=ax3,levels=np.arange(-8,8.1,0.5),cmap='BlueYellowRed',transform = ccrs.PlateCarree(),extend='both',
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Exposure (days)')
)
add_map_feature(ax3,cf_china, cf_9line)
(ds_ssp585_0std_gwl05_exposure_change['exposure']).plot(
    ax=ax3_si,levels=np.arange(-8,8.1,0.5),cmap='BlueYellowRed',transform = ccrs.PlateCarree(),extend='both',
    add_colorbar=False)
add_map_nanhai(ax3_si, cf_china, cf_9line)

ax4.cla()
(ds_ssp585_1std_gwl15_exposure_change['exposure']).plot(
    ax=ax4,levels=np.arange(-8,8.1,0.5),cmap='BlueYellowRed',transform = ccrs.PlateCarree(),extend='both',
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Exposure (days)')
)
add_map_feature(ax4,cf_china, cf_9line)
(ds_ssp585_1std_gwl15_exposure_change['exposure']).plot(
    ax=ax4_si,levels=np.arange(-8,8.1,0.5),cmap='BlueYellowRed',transform = ccrs.PlateCarree(),extend='both',
    add_colorbar=False
)
add_map_nanhai(ax4_si, cf_china, cf_9line)

ax5.cla()
(ds_ssp585_1std_gwl2_exposure_change['exposure']).plot(
    ax=ax5,levels=np.arange(-8,8.1,0.5),cmap='BlueYellowRed',transform = ccrs.PlateCarree(),extend='both',
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Exposure (days)')
)
add_map_feature(ax5,cf_china, cf_9line)
(ds_ssp585_1std_gwl2_exposure_change['exposure']).plot(
    ax=ax5_si,levels=np.arange(-8,8.1,0.5),cmap='BlueYellowRed',transform = ccrs.PlateCarree(),extend='both',
    add_colorbar=False)
add_map_nanhai(ax5_si, cf_china, cf_9line)

ax6.cla()
(ds_ssp585_1std_gwl05_exposure_change['exposure']).plot(
    ax=ax6,levels=np.arange(-8,8.1,0.5),cmap='BlueYellowRed',transform = ccrs.PlateCarree(),extend='both',
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Exposure (days)')
)
add_map_feature(ax6,cf_china, cf_9line)
(ds_ssp585_1std_gwl05_exposure_change['exposure']).plot(
    ax=ax6_si,levels=np.arange(-8,8.1,0.5),cmap='BlueYellowRed',transform = ccrs.PlateCarree(),extend='both',
    add_colorbar=False)
add_map_nanhai(ax6_si, cf_china, cf_9line)

ax1.text(0,1.03,'(a) GWL1.5-historical',transform=ax1.transAxes,fontsize=20)
ax2.text(0,1.03,'(b) GWL2.0-historical',transform=ax2.transAxes,fontsize=20)
ax3.text(0,1.03,'(c) GWL2.0-GWL1.5',transform=ax3.transAxes,fontsize=20)
ax4.text(0,1.03,'(d) GWL1.5-historical',transform=ax4.transAxes,fontsize=20)
ax5.text(0,1.03,'(e) GWL2.0-historical',transform=ax5.transAxes,fontsize=20)
ax6.text(0,1.03,'(f) GWL2.0-GWL1.5',transform=ax6.transAxes,fontsize=20)

ax1.text(0.65,1.03,'95% threshold',transform=ax1.transAxes,fontsize=20)
ax2.text(0.65,1.03,'95% threshold',transform=ax2.transAxes,fontsize=20)
ax3.text(0.65,1.03,'95% threshold',transform=ax3.transAxes,fontsize=20)
ax4.text(0.55,1.03,'95% threshold+1std',transform=ax4.transAxes,fontsize=20)
ax5.text(0.55,1.03,'95% threshold+1std',transform=ax5.transAxes,fontsize=20)
ax6.text(0.55,1.03,'95% threshold+1std',transform=ax6.transAxes,fontsize=20)

fig.show()
fig.savefig(fig_savepath+'figS_ssp585_exposure_diff.png',dpi=300)
