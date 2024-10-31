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
from matplotlib.colors import BoundaryNorm

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
from matplotlib.ticker import ScalarFormatter
import proplot as pplt
from matplotlib.patches import Patch

cf_china = cfeature.ShapelyFeature(
    Reader(r'/home/zq2/wg/code/shp/china.shp').geometries(),
    ccrs.PlateCarree(),
    edgecolor='gray',
    facecolor='none'
)
cf_9line = cfeature.ShapelyFeature(
    Reader(r'/home/zq2/wg/codeshp/9lines_hainan_taiwan.shp').geometries(),
    ccrs.PlateCarree(),
    edgecolor='gray',
    facecolor='none'
)

def load_region(file):
    # TODO: make variable name more generic
    return xr.open_dataset(file).source_region

region_border = load_region('/mnt/nas-1401-2/clusters_analysise_path/hws_metrics/china/mask/border_province.nc').rename({'latitude':'lat','longitude':'lon'})
region_china = load_region('/mnt/nas-1401-2/clusters_analysise_path/hws_metrics/china/mask/china.nc').rename({'latitude':'lat','longitude':'lon'})
region_coastal = load_region('/mnt/nas-1401-2/clusters_analysise_path/hws_metrics/china/mask/coastal_province.nc').rename({'latitude':'lat','longitude':'lon'})
region_inner = load_region('/mnt/nas-1401-2/clusters_analysise_path/hws_metrics/china/mask/inner_province.nc').rename({'latitude':'lat','longitude':'lon'})

def plot_contourf(ax, da, cmap,levels):

    m = ax.contourf(da.lon,da.lat,da.data,levels=levels,cmap=cmap, transform = ccrs.PlateCarree(),extend='max')
    # m = ax.contourf(da.lon,da.lat,da.data,levels = bounds,cmap=cmap_2, norm=norm,transform = ccrs.PlateCarree(),extend='neither')

    return m

def xarray_weighted_mean(data_array):

    weights = np.cos(np.deg2rad(data_array.lat))
    weights.name = "weights"
    data_weighted = data_array.weighted(weights)
    weighted_mean = data_weighted.mean(("lon", "lat"))

    return weighted_mean

def filter_2d_to_1d(data):
    filterd_data = data[~np.isnan(data)]

    return filterd_data

def mk_trend_ve(x):
    if np.isnan(x).sum() > 25:
        return (np.nan ,np.nan)
    else :
        mk_result = mk.original_test(x)
        slope = mk_result.slope
        p = mk_result.p
        return (slope ,p)
#%% data progress
# pr spatial
mean_sp_path = '/mnt/nas-1401-2/clusters_analysise_path/hws_metrics/china/attribution_detection_data/mean_spatial/'
xr_ex_hws_nat = xr.open_dataset(mean_sp_path + 'hist-nat_tm95pct_China_ocean_onto_land_hws_1979-2014.nc')
xr_ex_hws_ghg = xr.open_dataset(mean_sp_path + 'hist-GHG_tm95pct_China_ocean_onto_land_hws_1979-2014.nc')
xr_ex_hws_aer = xr.open_dataset(mean_sp_path + 'hist-aer_tm95pct_China_ocean_onto_land_hws_1979-2014.nc')
xr_ex_hws_his = xr.open_dataset(mean_sp_path + 'historical_tm95pct_China_ocean_onto_land_hws_1979-2014.nc')

xr_pr_ex_hws_ghg = xr_ex_hws_ghg/xr_ex_hws_nat
xr_pr_ex_hws_aer = xr_ex_hws_aer/xr_ex_hws_nat
xr_pr_ex_hws_his = xr_ex_hws_his/xr_ex_hws_nat

hws_metrics_path = '/media/zq2/Reanalysis_data/clusters_analysise_path/hws_metrics/china_obs/'
xr_ex_hws_obs = xr.open_dataset(hws_metrics_path+'ERA5_tm95pct_China_ocean_onto_land_hws_metrics_1979-2020.nc')

ts_path = '/mnt/nas-1401-2/clusters_analysise_path/hws_metrics/china/attribution_detection_data/yearly_ts/'
df_hws_deck = pd.read_csv(ts_path+'hws_ens_deck.csv')

#%%
fig_dir = '/mnt/nas-1401-2/clusters_analysise_path/hws_metrics/china/attribution_detection_data/plot/'

forcings_colors = ['red', 'blue', '#E040FB','green']
array = [  # the "picture" (1 == subplot A, 2 == subplot B, etc.)
    [1, 1, 2, 2, 3, 3],
    [1, 1, 2, 2, 3, 3],
    [1, 1, 2, 2, 3, 3],
    [4, 4, 4, 5, 5, 5],
    [4, 4, 4, 5, 5, 5],
    [6, 6, 7, 7, 8, 8],
    [6, 6, 7, 7, 8, 8]
]

proj_china=ccrs.LambertConformal(central_longitude=107.5,central_latitude=30.0,standard_parallels=(20,50))

fig, axs = pplt.subplots(array, figwidth=11, span=True,left='3em',proj={1: proj_china, 2: proj_china,3: proj_china},share=False,tight=True,panelpad='0em')
# fig, axs = pplt.subplots(array, figwidth=10, span=False,share=False,left='3em',)
axs.format(abc=True,grid=False,)

bounds1 = np.array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.4, 1.8, 2.2, 2.6, 3. , 3.4, 3.8, 4.2, 4.6, 5.])

# ax1
# m0 = plot_contourf(axs[0],xr_pr_ex_hws_his['exposure'].where(region_china, drop=True), cmap=cmaps.MPL_RdYlBu_r,levels=np.arange(0,5.1,0.2))
m0 = plot_contourf(axs[0],xr_pr_ex_hws_his['exposure'].where(region_china, drop=True), cmap=cmaps.BlueYellowRed,levels=bounds1)

# axs[0].format(lonlim=(80, 135), latlim=(15, 55))
axs[0].format(lonlim=(80, 130), latlim=(2, 55))
axs[0].add_feature(cf_china,edgecolor='black')
axs[0].add_feature(cf_9line,edgecolor='black')
axs[0].add_feature(cfeature.COASTLINE.with_scale('110m'),edgecolor='#626063')
lb1=axs[0].gridlines(draw_labels=False, xlocs=range(0,180,10), ylocs=range(0,90,10), linestyle=(0,(10,10)), linewidth=0.25, color='dimgrey', alpha=0.8, zorder=4)
lb1=axs[0].gridlines(draw_labels=True,x_inline=False, y_inline=False,xlocs=range(80,135,10), ylocs=range(10,51,10),linewidth=0.1, color='gray', alpha=0.8, linestyle='--' )
lb1.top_labels = None
lb1.right_labels = None
lb1.rotate_labels = False
axs[0].colorbar(m0,loc='b',label='Probability Ratio',fraction=0.02, shrink=0.9,pad=0.3,ticks=[0,0.5,1,3,5])
# axs[0].add_feature(cfeature.OCEAN, facecolor='#97DBF2')
# axs[0].add_feature(cfeature.LAND)
fig.show()
# def add_nanhai_ax(bbox):
#     ax_si = fig.add_axes([bbox.xmax-0.08, bbox.ymin, 0.093, 0.11], frameon=True, projection=proj_china)
#     ax_si.add_feature(cf_china, edgecolor='black')
#     ax_si.add_feature(cf_9line, edgecolor='black')
#     ax_si.add_feature(cfeature.COASTLINE.with_scale('110m'), edgecolor='#626063')
#     ax_si.set_extent([105.8, 122, 2, 25], crs=ccrs.PlateCarree())
#     return ax_si
#
# axs0_si = add_nanhai_ax(axs[0].get_position())
# m0_si = plot_contourf(axs0_si,xr_pr_ex_hws_his['exposure'].where(region_china, drop=True), cmap='Rocket')
# ax2
bounds2 = np.array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.])

m1 = plot_contourf(axs[1],xr_pr_ex_hws_ghg['exposure'].where(region_china, drop=True), cmap=cmaps.BlueYellowRed,levels=bounds2)
axs[1].format(lonlim=(80, 130), latlim=(2, 55))
axs[1].add_feature(cf_china,edgecolor='black')
axs[1].add_feature(cf_9line,edgecolor='black')
axs[1].add_feature(cfeature.COASTLINE.with_scale('110m'),edgecolor='#626063')
lb1=axs[1].gridlines(draw_labels=False, xlocs=range(0,180,10), ylocs=range(0,90,10), linestyle=(0,(10,10)), linewidth=0.25, color='dimgrey', alpha=0.8, zorder=4)
lb1=axs[1].gridlines(draw_labels=True,x_inline=False, y_inline=False,xlocs=range(80,135,10), ylocs=range(10,51,10),linewidth=0.1, color='gray', alpha=0.8, linestyle='--' )
lb1.top_labels = None
lb1.right_labels = None
lb1.rotate_labels = False
axs[1].colorbar(m1,loc='b',label='Probability Ratio',fraction=0.02, shrink=0.9,pad=0.3,ticks=[0,0.5,1,6,11])

# ax3
m2 = plot_contourf(axs[2],xr_pr_ex_hws_aer['exposure'].where(region_china, drop=True), cmap=cmaps.BlueYellowRed,levels=np.arange(0,2.01,0.05))
axs[2].format(lonlim=(80, 130), latlim=(2, 55))
axs[2].add_feature(cf_china,edgecolor='black')
axs[2].add_feature(cf_9line,edgecolor='black')
axs[2].add_feature(cfeature.COASTLINE.with_scale('110m'),edgecolor='#626063')
lb1=axs[2].gridlines(draw_labels=False, xlocs=range(0,180,10), ylocs=range(0,90,10), linestyle=(0,(10,10)), linewidth=0.25, color='dimgrey', alpha=0.8, zorder=4)
lb1=axs[2].gridlines(draw_labels=True,x_inline=False, y_inline=False,xlocs=range(80,135,10), ylocs=range(10,51,10),linewidth=0.1, color='gray', alpha=0.8, linestyle='--' )
lb1.top_labels = None
lb1.right_labels = None
lb1.rotate_labels = False
axs[2].colorbar(m2,loc='b',label='Probability Ratio',fraction=0.02, shrink=0.9,pad=0.3,ticks=np.arange(0,2.1,0.5))

fig.show()

# ax4 anomaly
def get_ts_ano(df):
    mean_value = df['mean'].mean()

    new_df = df.copy()
    new_df['mean'] = df['mean'] - mean_value
    new_df['low'] = df['low'] - mean_value
    new_df['high'] = df['high'] - mean_value

    return new_df
def select_df(df, region, deck, type, metric):
    filtered_df = df[(df['region'] == region) & (df['deck'] == deck) & (df['type'] == type) & (df['metric'] == metric)]

    return filtered_df

def get_ts_far(df_nat,df_deck):

    new_df = df_deck.copy()

    new_df['mean'] = 1- (df_nat['mean']/df_deck['mean'].values).values
    new_df['low'] = 1- (df_nat['low']/df_deck['mean'].values).values
    new_df['high'] = 1- (df_nat['high']/df_deck['mean'].values).values

    return new_df

# filter
df_ex_exposure_aer_china_ts = select_df(df_hws_deck,region='china',deck='hist-aer',type='ocean_onto_land',metric='exposure')
df_ex_exposure_ghg_china_ts = select_df(df_hws_deck,region='china',deck='hist-GHG',type='ocean_onto_land',metric='exposure')
df_ex_exposure_nat_china_ts = select_df(df_hws_deck,region='china',deck='hist-nat',type='ocean_onto_land',metric='exposure')
df_ex_exposure_his_china_ts = select_df(df_hws_deck,region='china',deck='historical',type='ocean_onto_land',metric='exposure')

# cal anomaly
df_ex_exposure_aer_china_ano_ts = get_ts_ano(df_ex_exposure_aer_china_ts)
df_ex_exposure_ghg_china_ano_ts = get_ts_ano(df_ex_exposure_ghg_china_ts)
df_ex_exposure_nat_china_ano_ts = get_ts_ano(df_ex_exposure_nat_china_ts)
df_ex_exposure_his_china_ano_ts = get_ts_ano(df_ex_exposure_his_china_ts)

axs[3].cla()
axs[3].fill_between(np.arange(1979,2015),df_ex_exposure_nat_china_ano_ts['low'],df_ex_exposure_nat_china_ano_ts['high'],color='green',alpha=0.1)
axs[3].fill_between(np.arange(1979,2015),df_ex_exposure_aer_china_ano_ts['low'],df_ex_exposure_aer_china_ano_ts['high'],color='#E040FB',alpha=0.1)
axs[3].fill_between(np.arange(1979,2015),df_ex_exposure_ghg_china_ano_ts['low'],df_ex_exposure_ghg_china_ano_ts['high'],color='red',alpha=0.1)
axs[3].fill_between(np.arange(1979,2015),df_ex_exposure_his_china_ano_ts['low'],df_ex_exposure_his_china_ano_ts['high'],color='blue',alpha=0.1)

axs[3].plot(np.arange(1979,2015),df_ex_exposure_ghg_china_ano_ts['mean'],color='red',alpha=0.8)
axs[3].plot(np.arange(1979,2015),df_ex_exposure_aer_china_ano_ts['mean'],color='#E040FB',alpha=0.8)
axs[3].plot(np.arange(1979,2015),df_ex_exposure_his_china_ano_ts['mean'],color='blue',alpha=0.8)
axs[3].plot(np.arange(1979,2015),df_ex_exposure_nat_china_ano_ts['mean'],color='green',alpha=0.8)
axs[3].set_ylabel('Exposure anomalies (days)')
# axs[3].format(ylim=(-2,2))

axs[3].text(0.02,0.9,s='ALL forcing',horizontalalignment='left',verticalalignment='bottom',transform = axs[3].transAxes,fontweight='bold',fontsize=10,c = forcings_colors[0])
axs[3].text(0.02,0.8,s='1.39days/10yr*',horizontalalignment='left',verticalalignment='bottom',transform = axs[3].transAxes,fontweight='bold',fontsize=10,c = forcings_colors[0])
axs[3].text(0.275,0.9,s='GHG-only',horizontalalignment='left',verticalalignment='bottom',transform = axs[3].transAxes,fontweight='bold',fontsize=10,c = forcings_colors[1])
axs[3].text(0.275,0.8,s='2.11days/10yr*',horizontalalignment='left',verticalalignment='bottom',transform = axs[3].transAxes,fontweight='bold',fontsize=10,c = forcings_colors[1])
axs[3].text(0.521,0.9,s='AER-only',horizontalalignment='left',verticalalignment='bottom',transform = axs[3].transAxes,fontweight='bold',fontsize=10,c = forcings_colors[2])
axs[3].text(0.521,0.8,s='0.05days/10yr*',horizontalalignment='left',verticalalignment='bottom',transform = axs[3].transAxes,fontweight='bold',fontsize=10,c = forcings_colors[2])
axs[3].text(0.779,0.9,s='NAT-only',horizontalalignment='left',verticalalignment='bottom',transform = axs[3].transAxes,fontweight='bold',fontsize=10,c = forcings_colors[3])
axs[3].text(0.779,0.8,s='0.08days/10yr*',horizontalalignment='left',verticalalignment='bottom',transform = axs[3].transAxes,fontweight='bold',fontsize=10,c = forcings_colors[3])

# ax5
# cal ratio
def get_ts_ratio(df_land,df_mis,df_ex):

    df_new = df_ex.copy()
    df_new['mean'] = df_ex['mean'].values/(df_land['mean'].values+df_mis['mean'].values+df_ex['mean'].values)
    df_new['low'] = df_ex['low'].values / (df_land['mean'].values + df_mis['mean'].values+df_ex['mean'].values)
    df_new['high'] = df_ex['high'].values / (df_land['mean'].values + df_mis['mean'].values+df_ex['mean'].values)

    return df_new

df_land_exposure_aer_china_ts = select_df(df_hws_deck,region='china',deck='hist-aer',type='land',metric='exposure')
df_land_exposure_ghg_china_ts = select_df(df_hws_deck,region='china',deck='hist-GHG',type='land',metric='exposure')
df_land_exposure_nat_china_ts = select_df(df_hws_deck,region='china',deck='hist-nat',type='land',metric='exposure')
df_land_exposure_his_china_ts = select_df(df_hws_deck,region='china',deck='historical',type='land',metric='exposure')

df_mis_exposure_aer_china_ts = select_df(df_hws_deck,region='china',deck='hist-aer',type='miscellaneous',metric='exposure')
df_mis_exposure_ghg_china_ts = select_df(df_hws_deck,region='china',deck='hist-GHG',type='miscellaneous',metric='exposure')
df_mis_exposure_nat_china_ts = select_df(df_hws_deck,region='china',deck='hist-nat',type='miscellaneous',metric='exposure')
df_mis_exposure_his_china_ts = select_df(df_hws_deck,region='china',deck='historical',type='miscellaneous',metric='exposure')

df_ex_exposure_aer_china_ratio_ts = get_ts_ratio(df_land=df_land_exposure_aer_china_ts,
                                                   df_mis=df_mis_exposure_aer_china_ts,
                                                   df_ex=df_ex_exposure_aer_china_ts)
df_ex_exposure_ghg_china_ratio_ts = get_ts_ratio(df_land=df_land_exposure_ghg_china_ts,
                                                   df_mis=df_mis_exposure_ghg_china_ts,
                                                   df_ex=df_ex_exposure_ghg_china_ts)
df_ex_exposure_nat_china_ratio_ts = get_ts_ratio(df_land=df_land_exposure_nat_china_ts,
                                                   df_mis=df_mis_exposure_nat_china_ts,
                                                   df_ex=df_ex_exposure_nat_china_ts)
df_ex_exposure_his_china_ratio_ts = get_ts_ratio(df_land=df_land_exposure_his_china_ts,
                                                   df_mis=df_mis_exposure_his_china_ts,
                                                   df_ex=df_ex_exposure_his_china_ts)
# df_ex_exposure_aer_china_far_ts = get_ts_far(df_ex_exposure_nat_china_ts, df_ex_exposure_aer_china_ts)
# df_ex_exposure_ghg_china_far_ts = get_ts_far(df_ex_exposure_nat_china_ts, df_ex_exposure_ghg_china_ts)
# df_ex_exposure_his_china_far_ts = get_ts_far(df_ex_exposure_nat_china_ts, df_ex_exposure_his_china_ts)

axs[4].cla()
fb_nat = axs[4].fill_between(np.arange(1979,2015),df_ex_exposure_nat_china_ratio_ts['low'],df_ex_exposure_nat_china_ratio_ts['high'],color='green',alpha=0.1)
fb_aer = axs[4].fill_between(np.arange(1979,2015),df_ex_exposure_aer_china_ratio_ts['low'],df_ex_exposure_aer_china_ratio_ts['high'],color='#E040FB',alpha=0.1)
fb_ghg = axs[4].fill_between(np.arange(1979,2015),df_ex_exposure_ghg_china_ratio_ts['low'],df_ex_exposure_ghg_china_ratio_ts['high'],color='red',alpha=0.1)
fb_his = axs[4].fill_between(np.arange(1979,2015),df_ex_exposure_his_china_ratio_ts['low'],df_ex_exposure_his_china_ratio_ts['high'],color='blue',alpha=0.1)
#TODO：添加obs
l_ghg = axs[4].plot(np.arange(1979,2015),df_ex_exposure_ghg_china_ratio_ts['mean'],color='red',alpha=0.8)
l_aer = axs[4].plot(np.arange(1979,2015),df_ex_exposure_aer_china_ratio_ts['mean'],color='#E040FB',alpha=0.8)
l_his = axs[4].plot(np.arange(1979,2015),df_ex_exposure_his_china_ratio_ts['mean'],color='blue',alpha=0.8)
l_nat = axs[4].plot(np.arange(1979,2015),df_ex_exposure_nat_china_ratio_ts['mean'],color='green',alpha=0.8)
axs[4].set_ylabel('Exposure anomalies (days)')
axs[4].format(ylim=(0,1))

# labels = ['ALL','GHG','AER','NAT']
# legend_elements = [
#     Patch(facecolor=forcings_colors[0], edgecolor=forcings_colors[0],label='ALL',alpha=0.5),
#     Patch(facecolor=forcings_colors[1], edgecolor=forcings_colors[1],label='GHG',alpha=0.5),
#     Patch(facecolor=forcings_colors[2], edgecolor=forcings_colors[2],label='AER',alpha=0.5),
#     Patch(facecolor=forcings_colors[3], edgecolor=forcings_colors[3],label='NAT',alpha=0.5)]
# handles = legend_elements
# axs[4].legend(handles,labels,frameon=False,ncol=2,fontsize = 'medium', fancybox = True, loc='upper right') #bbox_to_anchor=[1.01,0.7],

axs[4].legend([(l_his,fb_his),(l_ghg,fb_ghg),(l_aer,fb_aer),(l_nat,fb_nat)], ['ALL','GHG','AER','NAT'],ncol=4,fontsize=10,frameon=False)

fig.show()
# ax6
# ex_exposure_aer_china_1d = filter_2d_to_1d(xr_ex_hws_aer['exposure'].where(region_china, drop=True).data)
ex_exposure_ghg_china_1d = filter_2d_to_1d(xr_ex_hws_ghg['exposure'].where(region_china, drop=True).data)
ex_exposure_nat_china_1d = filter_2d_to_1d(xr_ex_hws_nat['exposure'].where(region_china, drop=True).data)
ex_exposure_his_china_1d = filter_2d_to_1d(xr_ex_hws_his['exposure'].where(region_china, drop=True).data)
ex_exposure_obs_china_1d = filter_2d_to_1d(
    xr_ex_hws_obs['exposure'].where(region_china, drop=True).mean(dim='time').data)

df_ex_exposure_ghg_china = pd.DataFrame(ex_exposure_ghg_china_1d, columns=np.array([0.5]))
df_ex_exposure_nat_china = pd.DataFrame(ex_exposure_nat_china_1d, columns=np.array([0.6]))
df_ex_exposure_his_china = pd.DataFrame(ex_exposure_his_china_1d, columns=np.array([0.65]))
df_ex_exposure_obs_china = pd.DataFrame(ex_exposure_obs_china_1d, columns=np.array([0.7]))

axs[5].cla()
sns.distplot(pd.Series(ex_exposure_nat_china_1d),ax=axs[5],kde=True,color='green',hist_kws={'alpha':0.25})
sns.distplot(pd.Series(ex_exposure_obs_china_1d),ax=axs[5],kde=True, color='blue',hist_kws={'alpha':0.25})
sns.distplot(pd.Series(ex_exposure_ghg_china_1d),ax=axs[5],kde=True, color='red',hist_kws={'alpha':0.25})
sns.kdeplot(pd.Series(ex_exposure_his_china_1d),ax=axs[5], color='black')

axs[5].format(ylim=(0,0.8), xlim=(0,15))

# plot boxplot
al_axs5 = axs[5].alty(ylim=(0,0.8))
al_axs5.boxplot(df_ex_exposure_ghg_china,vert=False,widths=0.03,showfliers=False,boxcolor='red',whiskercolor='red',capcolor='red',mediancolor='red')
al_axs5.boxplot(df_ex_exposure_nat_china,vert=False,widths=0.03,showfliers=False,boxcolor='green',whiskercolor='green',capcolor='green',mediancolor='green')
al_axs5.boxplot(df_ex_exposure_his_china,vert=False,widths=0.03,showfliers=False,boxcolor='blue',whiskercolor='blue',capcolor='blue',mediancolor='blue')
al_axs5.boxplot(df_ex_exposure_obs_china,vert=False,widths=0.03,showfliers=False,boxcolor='black',whiskercolor='black',capcolor='black',mediancolor='black')

axs[5].text(s='Hist-ALL',x = 6.8, y=0.7,fontdict={'fontsize':11,'color':'blue','fontweight':'medium'})
axs[5].text(s='GHG-only',x = 6.8, y=0.6,fontdict={'fontsize':11,'color':'red','fontweight':'medium'})
axs[5].text(s='NAT-only',x = 10.8, y=0.7,fontdict={'fontsize':11,'color':'green','fontweight':'medium'})
axs[5].text(s='ALL',x = 10.8, y=0.6,fontdict={'fontsize':11,'color':'black','fontweight':'medium'})

al_axs5.format(ylocator='null')

fig.show()
# ax7
ROF_OF_result_path = '/mnt/nas-1401-2/clusters_analysise_path/hws_metrics/china/attribution_detection_data/ROF_OF/'
df_sf_2sig = pd.read_csv(ROF_OF_result_path+'trends_scaling_factors_2signal.csv')
df_sf_3sig = pd.read_csv(ROF_OF_result_path+'trends_scaling_factors_3signal.csv')

color_re = ['#328cc3','#33af3d','#f41c26']
marker_re = ['^','o','*']

axs[6].cla()
axs[6].axhline(y=0,color='black',alpha=0.6,linewidth=0.8)
axs[6].axhline(y=1,linestyle=':',color='black',alpha=0.6,linewidth=0.8)
l1 = axs[6].plot((1,1),(df_sf_2sig['sf_min'][df_sf_2sig['forcing']=='ANT'],df_sf_2sig['sf_max'][df_sf_2sig['forcing']=='ANT']),color=color_re[0],alpha=0.8,linewidth=1.6)
axs[6].plot((2,2),(df_sf_2sig['sf_min'][df_sf_2sig['forcing']=='NAT'],df_sf_2sig['sf_max'][df_sf_2sig['forcing']=='NAT']),color=color_re[0],alpha=0.8,linewidth=1.6)
s1 = axs[6].scatter(x=1,y=df_sf_2sig['sf_best'][df_sf_2sig['forcing']=='ANT'],marker=marker_re[0],color=color_re[0])
axs[6].scatter(x=2,y=df_sf_2sig['sf_best'][df_sf_2sig['forcing']=='NAT'],marker=marker_re[0],color=color_re[0])
axs[6].set(ylim=(-1,3))
# axs[6].set(xlim=(0.4,2.6))
axs[6].set(xlim=(0.4,6.6))

# labels = ['','ANT','NAT','']
labels = ['','ANT','NAT','','GHG','NAT','AER','']
axs[6].set_xticklabels(labels)
axs[6].xaxis.set_tick_params(which='minor', bottom=False) # turn off xaxis minor ticks
# ax[d * 3].minorticks_off()
axs[6].set_ylabel('Scaling factor',fontdict={'fontsize':10})

#3signal
axs[6].axvline(x=3, color='black',linewidth=0.8)
axs[6].plot((4, 4),
                   (df_sf_3sig['sf_min'][df_sf_3sig['forcing'] == 'GHG'], df_sf_3sig['sf_max'][df_sf_3sig['forcing'] == 'GHG']),
                   color=color_re[0], alpha=0.8, linewidth=1.6)
axs[6].plot((5, 5),
                   (df_sf_3sig['sf_min'][df_sf_3sig['forcing'] == 'NAT'], df_sf_3sig['sf_max'][df_sf_3sig['forcing'] == 'NAT']),
                   color=color_re[0], alpha=0.8, linewidth=1.6)
axs[6].plot((6, 6),
                   (df_sf_3sig['sf_min'][df_sf_3sig['forcing'] == 'AER'], df_sf_3sig['sf_max'][df_sf_3sig['forcing'] == 'AER']),
                   color=color_re[0], alpha=0.8, linewidth=1.6)
axs[6].scatter(x=4, y=df_sf_3sig['sf_best'][df_sf_3sig['forcing'] == 'GHG'], marker=marker_re[0], color=color_re[0])
axs[6].scatter(x=5, y=df_sf_3sig['sf_best'][df_sf_3sig['forcing'] == 'NAT'], marker=marker_re[0], color=color_re[0])
axs[6].scatter(x=6, y=df_sf_3sig['sf_best'][df_sf_3sig['forcing'] == 'AER'], marker=marker_re[0], color=color_re[0])

# ax8
df_sf_3sig_ribes = pd.read_csv(ROF_OF_result_path+'trends_scaling_factors_Ribes.csv')
color_f = ['#abc9ea','#efb792','#98daa7','#f3aba8']

err1_obs = df_sf_3sig_ribes['trend'][df_sf_3sig_ribes['forcing'] == 'Observation'] - df_sf_3sig_ribes['trend_min'][df_sf_3sig_ribes['forcing'] == 'Observation']
err2_obs = df_sf_3sig_ribes['trend_max'][df_sf_3sig_ribes['forcing'] == 'Observation'] - df_sf_3sig_ribes['trend'][df_sf_3sig_ribes['forcing'] == 'Observation']
err1_ghg = df_sf_3sig_ribes['trend'][df_sf_3sig_ribes['forcing'] == 'Forcing no 1 only'] - df_sf_3sig_ribes['trend_min'][
    df_sf_3sig_ribes['forcing'] == 'Forcing no 1 only']
err2_ghg = df_sf_3sig_ribes['trend_max'][df_sf_3sig_ribes['forcing'] == 'Forcing no 1 only'] - df_sf_3sig_ribes['trend'][
    df_sf_3sig_ribes['forcing'] == 'Forcing no 1 only']
err1_nat = df_sf_3sig_ribes['trend'][df_sf_3sig_ribes['forcing'] == 'Forcing no 2 only'] - df_sf_3sig_ribes['trend_min'][
    df_sf_3sig_ribes['forcing'] == 'Forcing no 2 only']
err2_nat = df_sf_3sig_ribes['trend_max'][df_sf_3sig_ribes['forcing'] == 'Forcing no 2 only'] - df_sf_3sig_ribes['trend'][
    df_sf_3sig_ribes['forcing'] == 'Forcing no 2 only']
err1_aer = df_sf_3sig_ribes['trend'][df_sf_3sig_ribes['forcing'] == 'Forcing no 3 only'] - df_sf_3sig_ribes['trend_min'][
    df_sf_3sig_ribes['forcing'] == 'Forcing no 3 only']
err2_aer = df_sf_3sig_ribes['trend_max'][df_sf_3sig_ribes['forcing'] == 'Forcing no 3 only'] - df_sf_3sig_ribes['trend'][
    df_sf_3sig_ribes['forcing'] == 'Forcing no 3 only']

axs[7].axhline(y=0, color='black', alpha=0.6, linewidth=0.8)
p1 = axs[7].bar(1, df_sf_3sig_ribes['trend'][df_sf_3sig_ribes['forcing'] == 'Observation'],
                  yerr=[[err1_obs.values[0]], [err2_obs.values[0]]], align='center', edgecolor=color_f[0],
                  facecolor=color_f[0], width=0.8, error_kw=dict(lw=0.8, capsize=3, capthick=0.8))
p2 = axs[7].bar(2, df_sf_3sig_ribes['trend'][df_sf_3sig_ribes['forcing'] == 'Forcing no 1 only'],
                  yerr=[[err1_ghg.values[0]], [err2_ghg.values[0]]], align='center', edgecolor=color_f[1],
                  facecolor=color_f[1], width=0.8, error_kw=dict(lw=0.5, capsize=3, capthick=0.8))
p3 = axs[7].bar(3, df_sf_3sig_ribes['trend'][df_sf_3sig_ribes['forcing'] == 'Forcing no 2 only'],
                  yerr=[[err1_nat.values[0]], [err2_nat.values[0]]], align='center', edgecolor=color_f[2],
                  facecolor=color_f[2], width=0.8, error_kw=dict(lw=0.5, capsize=3, capthick=0.8))
p4 = axs[7].bar(4, df_sf_3sig_ribes['trend'][df_sf_3sig_ribes['forcing'] == 'Forcing no 3 only'],
                  yerr=[[err1_aer.values[0]], [err2_aer.values[0]]], align='center', edgecolor=color_f[3],
                  facecolor=color_f[3], width=0.8, error_kw=dict(lw=0.5, capsize=3, capthick=0.8))
axs[7].set(xlim=(0.2, 4.8))
labels = ['OBS', 'GHG', 'NAT', 'AER']
axs[7].set_xticks([1, 2, 3, 4])
axs[7].set_xticklabels(labels, fontdict={'fontsize': 9})
axs[7].xaxis.set_tick_params(which='minor', bottom=False)  # turn off xaxis minor ticks
axs[7].set_ylabel('Trend (m/decade)', fontdict={'fontsize': 10})
axs[7].grid(False)



fig.save(fig_dir + '16.pdf')
fig.save(fig_dir + '1.png')
