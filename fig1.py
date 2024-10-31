#%%
import numpy as np
from netCDF4 import Dataset
import xarray as xr
from datetime import datetime
import pandas as pd
import pickle
import pymannkendall as mk
import scipy.stats as stats
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
import seaborn as sns
import matplotlib.transforms as mtransforms

def mk_trend_ve(x):
    if np.isnan(x).sum() > 25:
        return (np.nan ,np.nan)
    else :
        mk_result = mk.original_test(x)
        slope = mk_result.slope
        p = mk_result.p
        return (slope ,p)

def xarray_weighted_mean(data_array):

    weights = np.cos(np.deg2rad(data_array.lat))
    weights.name = "weights"
    data_weighted = data_array.weighted(weights)
    weighted_mean = data_weighted.mean(("lon", "lat"))

    return weighted_mean

def add_map_nanhai(ax,cf_china,cf_9line):

    ax.add_feature(cf_china, edgecolor='black')
    ax.add_feature(cf_9line, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), edgecolor='#626063')
    ax.set_extent([105.8,122,2,25], crs=ccrs.PlateCarree())  # nanhai

def add_map_feature(ax, cf_china, cf_9line):
    # ax.add_feature(cfeature.COASTLINE.with_scale('110m'), edgecolor='#626063')
    # # ax.add_feature(cf_coastial_land_200km)
    # ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    # ax.set_yticks(np.arange(-60, 90, 30), crs=ccrs.PlateCarree())
    # ax.xaxis.set_major_formatter(LongitudeFormatter())
    # ax.yaxis.set_major_formatter(LatitudeFormatter())
    # ax.set_extent([-180, 180, -65, 90], crs=ccrs.PlateCarree())
    # ax.axes.set_xlabel('')
    # ax.axes.set_ylabel('')

    ax.add_feature(cf_china, edgecolor='black')
    ax.add_feature(cf_9line, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), edgecolor='#626063')
    lb = ax.gridlines(draw_labels=False, xlocs=range(0, 180, 10), ylocs=range(0, 90, 10), linestyle=(0, (10, 10)),
                        linewidth=0.25, color='dimgrey', alpha=0.8, zorder=4)
    lb = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, xlocs=range(80, 135, 10),
                        ylocs=range(0, 51, 10), linewidth=0.1, color='gray', alpha=0.8, linestyle='--')
    lb.top_labels = None
    lb.right_labels = None
    lb.left_labels = None
    lb.rotate_labels = False

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = '19'

cf_china = cfeature.ShapelyFeature(
    Reader(r'/home/zq2/wg/code/shp/china.shp').geometries(),
    ccrs.PlateCarree(),
    edgecolor='gray',
    facecolor='none'
)
cf_9line = cfeature.ShapelyFeature(
    Reader(r'/home/zq2/wg/code/shp/9lines_hainan_taiwan.shp').geometries(),
    ccrs.PlateCarree(),
    edgecolor='gray',
    facecolor='none'
)

da_china_mask = xr.open_dataarray('/home/zq2/wg/code/global_landfalling_heatwaves/src/plot/china_mask.nc')
da_china_mask = da_china_mask.rename({'longitude': 'lon', 'latitude': 'lat'})

# read different type of hws metrics
hws_metrics_path = '/media/zq2/Reanalysis_data/clusters_analysise_path/hws_metrics/china_obs/'
ds_ocean_onto_land_hws = xr.open_dataset(hws_metrics_path+'ERA5_tm95pct_China_ocean_onto_land_hws_metrics_1979-2020.nc')
# ds_ocean_onto_land_hws = xr.open_dataset(hws_metrics_path+'ERA5_tm90pct_median_10000_0.7_withAQ_ocean_onto_land_hws_metrics_1979-2020.nc')


# exposure ratio
hws_metrics = 'exposure'
da_exposure_ratio = ds_ocean_onto_land_hws[hws_metrics].where(da_china_mask, drop=True).sum(dim='time')/(
    ds_land_hws[hws_metrics].where(da_china_mask, drop=True).sum(dim='time')+
    ds_miscellaneous_hws[hws_metrics].where(da_china_mask, drop=True).sum(dim='time')+
    ds_ocean_onto_land_hws[hws_metrics].where(da_china_mask, drop=True).sum(dim='time')
)
lons = da_exposure_ratio.lon.data
lats = da_exposure_ratio.lat.data

data_lm_trend = xr.apply_ufunc(
    mk_trend_ve,
    ds_ocean_onto_land_hws[hws_metrics].where(da_china_mask, drop=True),
    input_core_dims = [['time']],
    output_core_dims = [[],[]],
    vectorize=True
)

# fig1
fig = plt.figure(figsize=[18,18])
proj_china=ccrs.LambertConformal(central_longitude=107.5,central_latitude=30.0,standard_parallels=(20,50))
ax1 = fig.add_axes([0.075, 0.75, 0.24, 0.14], frameon=True)
ax2 = fig.add_axes([0.4, 0.75, 0.58, 0.14], frameon=True)

# ax3 = fig.add_axes([0.075, 0.45, 0.88, 0.31], frameon=True,projection=ccrs.PlateCarree(central_longitude =150.0))#上1

ax3 = fig.add_axes([0.05, 0.43, 0.29, 0.4], frameon=True,projection=proj_china) #中1
ax3_si = fig.add_axes([0.271, 0.494, 0.086, 0.07], frameon=True,projection=proj_china)
ax4 = fig.add_axes([0.37, 0.43, 0.29, 0.4], frameon=True,projection=proj_china) #中2
ax4_si = fig.add_axes([0.591, 0.494, 0.086, 0.07], frameon=True,projection=proj_china)
ax5 = fig.add_axes([0.69, 0.43, 0.29, 0.4], frameon=True,projection=proj_china) #中2
ax5_si = fig.add_axes([0.911, 0.494, 0.086, 0.07], frameon=True,projection=proj_china)

ax6 = fig.add_axes([0.05, 0.1, 0.29, 0.4], frameon=True,projection=proj_china) #中1
ax6_si = fig.add_axes([0.271, 0.164, 0.086, 0.07], frameon=True,projection=proj_china)
ax7 = fig.add_axes([0.37, 0.1, 0.29, 0.4], frameon=True,projection=proj_china) #中2
ax7_si = fig.add_axes([0.591, 0.164, 0.086, 0.07], frameon=True,projection=proj_china)
ax8 = fig.add_axes([0.69, 0.1, 0.29, 0.4], frameon=True,projection=proj_china) #中2
ax8_si = fig.add_axes([0.911, 0.164, 0.086, 0.07], frameon=True,projection=proj_china)

bbox = ax3.get_position()

extent_nanhai=[105.8, 122,0,25]

# (a)
ax1.cla()
total_hws_event_list = [total_ocean_onto_land_hws,
                        total_miscellaneous_hws,
                        total_land_hws]
pie_labels = ['Exogenous', 'Other', 'Endogenous']
ax1.pie(x=total_hws_event_list,labels=pie_labels,
          wedgeprops={'width':0.5},
          textprops={'fontsize':20},
          colors=['#f18c54','#c5d6f0','#a9ca70'])
# colors=['#67D5B5','#EE7785','#C89EC4'] 原
# colors=['#f18c54','#c5d6f0','#a9ca70']
ax1.text(0.85,0.75,'(42.7%)',transform=ax1.transAxes,fontsize=20) #ex
ax1.text(0.93,-0.03,'(29.9%)',transform=ax1.transAxes,fontsize=20) #en
ax1.text(-0.16,0.14,'(27.4%)',transform=ax1.transAxes,fontsize=20) #other

# (b)
ax2.cla()
years = list(range(1979, 2021))
event_num_landfalling = np.array(event_num_summary['ocean_onto_land'].tolist())
# ax2.bar(years, event_num_landfalling, color='#ff7473') 
# ax2.plot(years, event_num_landfalling, color='#CE6D39')
# ax2.xlabel('Year')
# ax2.ylabel('Events')
norm = plt.Normalize(0, 33)
sm = plt.cm.ScalarMappable(cmap="RdYlBu_r", norm=norm)
colors = sm.to_rgba(event_num_landfalling)
sns.barplot(x=np.arange(1979, 2021), y=event_num_landfalling, palette=colors, ax=ax2)
ax2.set_xticks([1,11,21,31,41])
slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(0,42),event_num_landfalling)
# 
regression_line = slope * np.arange(0,42) + intercept
ax2.plot(np.arange(0,42), regression_line, color='black',linestyle='--',linewidth = '2')

fig.show()

#（c）exposure 
ds_ocean_onto_land_hws['exposure'].mean(dim='time').where(da_china_mask, drop=True).plot(
    ax=ax3,levels=np.arange(0,6.4,0.2),cmap=cmaps.MPL_Oranges,transform = ccrs.PlateCarree(),extend='max',
    cbar_kwargs=dict(fraction=0.06, shrink=0.85,orientation='horizontal',label='Exposure (days)',pad=0.1)
)
ax3.add_feature(cf_china,edgecolor='black')
ax3.add_feature(cf_9line,edgecolor='black')
ax3.add_feature(cfeature.COASTLINE.with_scale('110m'),edgecolor='#626063')
# ax4.add_feature(cfeature.LAND)
lb3=ax3.gridlines(draw_labels=False, xlocs=range(0,180,10), ylocs=range(0,90,10), linestyle=(0,(10,10)), linewidth=0.25, color='dimgrey', alpha=0.8, zorder=4)
lb3=ax3.gridlines(draw_labels=True,x_inline=False, y_inline=False,xlocs=range(80,135,10), ylocs=range(20,51,10),linewidth=0.1, color='gray', alpha=0.8, linestyle='--' )
lb3.top_labels = None
lb3.right_labels = None
lb3.rotate_labels = False

ds_ocean_onto_land_hws['exposure'].mean(dim='time').where(da_china_mask, drop=True).plot(
    ax=ax3_si,levels=np.arange(0,6.4,0.2),cmap=cmaps.MPL_Oranges,transform = ccrs.PlateCarree(),extend='max',
    add_colorbar=False
)
add_map_nanhai(ax3_si, cf_china, cf_9line)

# （d）ratio mean
(da_exposure_ratio.where(da_china_mask, drop=True)*100).plot(
    ax=ax4,levels=np.arange(0,101,5),cmap='Spectral_r',transform = ccrs.PlateCarree(),extend='neither',
        cbar_kwargs=dict(fraction=0.06, shrink=0.85,orientation='horizontal',label='Ratio (%)',pad=0.1)
)
add_map_feature(ax4, cf_china, cf_9line)

(da_exposure_ratio.where(da_china_mask, drop=True)*100).plot(
    ax=ax4_si,levels=np.arange(0,101,5),cmap='Spectral_r',transform = ccrs.PlateCarree(),extend='neither',
    add_colorbar=False
)
add_map_nanhai(ax4_si, cf_china, cf_9line)

# （e）frequency mean:
ds_ocean_onto_land_hws['frequency'].mean(dim='time',skipna=True).where(da_china_mask, drop=True).plot(
    ax=ax5,levels=np.arange(0,2.2,0.2),cmap='MPL_Oranges',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.85,orientation='horizontal',label='Frequency (events per year)',pad=0.1)
)
add_map_feature(ax5, cf_china, cf_9line)
ds_ocean_onto_land_hws['frequency'].mean(dim='time',skipna=True).where(da_china_mask, drop=True).plot(
    ax=ax5_si,levels=np.arange(0,2.2,0.2),cmap='MPL_Oranges',transform = ccrs.PlateCarree(),
    add_colorbar=False
)
add_map_nanhai(ax5_si, cf_china, cf_9line)

# ax6
ax6.cla()
ds_ocean_onto_land_hws['extent'].mean(dim='time',skipna=True).where(da_china_mask, drop=True).plot(
    ax=ax6,levels=np.arange(200000,4000000,800000),cmap=cmaps.MPL_copper_r,transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.85,orientation='horizontal',label='Extent (km$^2$)',pad=0.1)
)
ax6.add_feature(cf_china,edgecolor='black')
ax6.add_feature(cf_9line,edgecolor='black')
ax6.add_feature(cfeature.COASTLINE.with_scale('110m'),edgecolor='#626063')
# ax4.add_feature(cfeature.LAND)
lb6=ax6.gridlines(draw_labels=False, xlocs=range(0,180,10), ylocs=range(0,90,10), linestyle=(0,(10,10)), linewidth=0.25, color='dimgrey', alpha=0.8, zorder=4)
lb6=ax6.gridlines(draw_labels=True,x_inline=False, y_inline=False,xlocs=range(80,135,10), ylocs=range(20,51,10),linewidth=0.1, color='gray', alpha=0.8, linestyle='--' )
lb6.top_labels = None
lb6.right_labels = None
lb6.rotate_labels = False

ds_ocean_onto_land_hws['extent'].mean(dim='time',skipna=True).where(da_china_mask, drop=True).plot(
    ax=ax6_si,levels=np.arange(400000,4000000,800000),cmap=cmaps.MPL_copper_r,transform = ccrs.PlateCarree(),
    add_colorbar=False
)
add_map_nanhai(ax6_si, cf_china, cf_9line)

# ax7
ax7.cla()
ds_ocean_onto_land_hws['cum_heat_grid'].mean(dim='time',skipna=True).where(da_china_mask, drop=True).plot(
    ax=ax7,levels=np.arange(0,8.1,0.8),cmap=cmaps.cmocean_matter,transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.85,orientation='horizontal',label='Cumulative Heat (°C)',pad=0.1)
)
add_map_feature(ax7, cf_china, cf_9line)
ds_ocean_onto_land_hws['cum_heat_grid'].mean(dim='time',skipna=True).where(da_china_mask, drop=True).plot(
    ax=ax7_si,levels=np.arange(0,8.1,0.8),cmap=cmaps.cmocean_matter,transform = ccrs.PlateCarree(),
    add_colorbar=False
)
add_map_nanhai(ax7_si, cf_china, cf_9line)

# ax8
ax8.cla()
(ds_ocean_onto_land_hws['cum_heat_grid'].mean(dim='time',skipna=True).where(da_china_mask, drop=True)/\
ds_ocean_onto_land_hws['exposure'].mean(dim='time',skipna=True).where(da_china_mask, drop=True)).plot(
    ax=ax8,levels=np.arange(0,2.2,0.2),cmap=cmaps.cmocean_matter,transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.85,orientation='horizontal',label='Intensity (°C/day)',pad=0.1)
)
add_map_feature(ax8, cf_china, cf_9line)
(ds_ocean_onto_land_hws['cum_heat_grid'].mean(dim='time',skipna=True).where(da_china_mask, drop=True)/\
ds_ocean_onto_land_hws['exposure'].mean(dim='time',skipna=True).where(da_china_mask, drop=True)).plot(
    ax=ax8_si,levels=np.arange(0,2.2,0.2),cmap=cmaps.cmocean_matter,transform = ccrs.PlateCarree(),
    add_colorbar=False
)
add_map_nanhai(ax8_si, cf_china, cf_9line)

ax1.text(-0.53,1.03,'a',fontweight='bold',transform=ax1.transAxes,fontsize=25)
ax2.text(0,1.03,'b',fontweight='bold',transform=ax2.transAxes,fontsize=25)
ax3.text(0,1.03,'c',fontweight='bold',transform=ax3.transAxes,fontsize=25)
ax4.text(0,1.03,'d',fontweight='bold',transform=ax4.transAxes,fontsize=25)
ax5.text(0,1.03,'e',fontweight='bold',transform=ax5.transAxes,fontsize=25)
ax6.text(0,1.03,'f',fontweight='bold',transform=ax6.transAxes,fontsize=25)
ax7.text(0,1.03,'g',fontweight='bold',transform=ax7.transAxes,fontsize=25)
ax8.text(0,1.03,'h',fontweight='bold',transform=ax8.transAxes,fontsize=25)

ax2.text(0.01,0.87,'unit: events',transform=ax2.transAxes,fontsize=22)
# ax3.text(0.01,0.05,'Mean: 3.49 days',transform=ax3.transAxes,fontsize=24)
# ax4.text(0.01,0.05,'Mean: 54.87%',transform=ax4.transAxes,fontsize=24)
# ax5.text(0.01,0.05,'Mean: 1.05days/10yr',transform=ax5.transAxes,fontsize=24)

fig.show()



fig.show()
fig_savepath = '/home/zq2/wg/code/src/plot/china/part1/figure/'
fig.savefig(fig_savepath + 'fig1_1030.pdf',bbox_inches='tight')
fig.savefig(fig_savepath+'Fig1_1030.png',dpi=600)