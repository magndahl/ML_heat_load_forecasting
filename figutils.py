# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 16:12:39 2017

@author: azfv1n8
"""

import pandas as pd
import numpy as np
import datetime as dt
import cPickle as pickle
import calendar
from collections import OrderedDict

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity


from custom_metrics import root_mean_squared_error, mean_absolute_percentage_error, mean_error
from test_perfect_wfc import prepare_cv_test_data, test_scenarios, calc_scenario_rescalced_metrics 
ocean_five = {'OLS':'#CC333F', 'MLP':'#EDC951', 'SVR':'#00A0B0'}
darkgrey = '#333333'

plt.close('all')
text_width = 6.299 #in

savepath = 'figures/articlefigs/'
scenarios = ['Sc%i'%i for i in (1,2,3)]
models = ['OLS', 'MLP', 'SVR']


sc_text = {'Sc1':'Only weather data',
           'Sc2':'Weather and calendar',
           'Sc3': 'Weather, calendar\nand holidays'}





#figure 1
def plot_example_year():
    cv_df = pd.read_pickle('data/cleaned/assembled_data/cv_data.pkl')
    fig, ax = plt.subplots(figsize=(text_width, 0.67*text_width))
    ts_start = dt.datetime(2010,1,1,1)
    ts_end = dt.datetime(2011,1,1,0)
    zoom_start = dt.datetime(2010,3,8,1)
    zoom_end = dt.datetime(2010,3,15,0)

    prod = cv_df.loc[ts_start:ts_end, 'prod']
    ax.plot(prod, lw=0.5)
       # axins = plt.axes([.42, .7, .3, .2])

    axins = plt.axes([.42, .6, .3, .32])
    axins.plot(prod[zoom_start:zoom_end])
    zoom_tick_pos = [zoom_start + dt.timedelta(days=i) for i in range(8)]
    axins.set_xticks(zoom_tick_pos)
    axins.xaxis.set_ticklabels([t.strftime('%a\n%b %#d') \
                                if t.weekday() in (0,6) else t.strftime('%a') \
                                for t in zoom_tick_pos], \
                                ha='left', fontsize=7, zorder=4)
    axins.set_yticks([500, 600, 700, 800])
    axins.yaxis.set_tick_params(labelsize=7)
    axins.set_xlim(zoom_start, zoom_end)
    axins.grid('on', linestyle=':')
    axins.set_ylabel('[MW]', fontsize=7)
                                           
    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="k", lw=0.5, zorder=3)
    ax.grid('on', linestyle=':')
    ax.set_xlim(ts_start, ts_end)
    ax.set_ylim(0,1050)
    xtick_pos = [dt.datetime(2010, i, 1,1) for i in range(1,13)]
    ax.set_xticks(xtick_pos)
    ax.xaxis.set_ticklabels([t.strftime('%b') for t in xtick_pos], ha='left')
    ax.set_xlabel('Example year 2010')
    ax.set_ylabel('Hourly heat load [MW]')
    
    fig.savefig(savepath + 'example_year.png', dpi=400)

    return fig


# figure 3
def plot_example_forecast():
    with open('data/results/test_perfect_wfc_preds.pkl', 'r') as f:
        preds_dict = pickle.load(f)
    
    cv_df = pd.read_pickle('data/cleaned/assembled_data/cv_data.pkl')
    test_df = pd.read_pickle('data/cleaned/assembled_data/test_data_real_fc.pkl')
    cv_Xs, cv_ys, test_Xs, test_ys, scalers = prepare_cv_test_data(scenarios, cv_df, test_df)
    
    ytrue = test_ys['Sc3']
    true_prod = pd.Series(scalers['Sc3'].inverse_transform_y(ytrue), index=test_df.index)
    pred_day_start = pd.datetime(2016,5,4,1)
    pred_day_end = pd.datetime(2016,5,5,0)
    xmin = pd.datetime(2016,5,3,1)
    xmax = pred_day_end
    ymin = 0
    ymax = 450
    pred_time = pred_day_start + dt.timedelta(hours=-15)
    xtickpos = [xmin + dt.timedelta(hours=6*i) for i in range(8)] + [xmax]
    
    sc = 'Sc3'
    fig, ax = plt.subplots(figsize=(text_width, 0.67*text_width))
    fclines = []
    for model in models:
        prod_pred = pd.Series(scalers[sc].inverse_transform_y(preds_dict[sc][model]), index=test_df.index)
        fclines.append(ax.plot(prod_pred[pred_day_start:pred_day_end], label=model+' forecast')[0])
        
    real_line, = ax.plot(true_prod[xmin:xmax], color=darkgrey, linestyle=':', label='Realized heat load')
    prod_for_fc, = ax.plot(true_prod[xmin:pred_time], color=darkgrey, linestyle='-', label='Heat load used for forecast')
    ax.vlines(pred_time, ymin, ymax, 'grey', lw=0.5)
    ax.vlines(pred_day_start, ymin, ymax, 'grey', lw=0.5)
    ax.grid('on', linestyle=':')
    ax.annotate('Time of forecast', xy=(pred_time, 375),  xytext=(pred_time + dt.timedelta(hours=8), 400),
                arrowprops=dict(arrowstyle='-|>', facecolor='black'),
                horizontalalignment='center', verticalalignment='bottom')
    ax.annotate('Beginning of\nforecasted day', xy=(pred_day_start, 225),  xytext=(pred_day_start + dt.timedelta(hours=2.5), 150),
                arrowprops=dict(arrowstyle='-|>', facecolor='black'),
                horizontalalignment='left', verticalalignment='bottom')
    
    
    ax.xaxis.set_ticks(xtickpos)
    ax.xaxis.set_ticklabels([t.strftime('%H:%M\n%b %#d') if t.hour==13 else t.strftime('%H:%M') for t in xtickpos])
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin,ymax)
    leg1 = ax.legend(fclines, ('OLS forecast', 'MLP forecast', 'SVR forecast'), loc=4)
    plt.gca().add_artist(leg1)
    leg2 = ax.legend([real_line, prod_for_fc], ('Realised heat load', 'Heat load used for forecast'), loc=3)
    
    ax.set_ylabel('Heat load [MW]')
    fig.tight_layout()
    fig.savefig(savepath + 'example_forecast.png', dpi=400)

    return fig

# figure 4
def perfect_real_fc_barplot():
    bin_width = 0.2
    
    cv_df = pd.read_pickle('data/cleaned/assembled_data/cv_data.pkl')
    test_df = pd.read_pickle('data/cleaned/assembled_data/test_data_real_fc.pkl')
    cv_Xs, cv_ys, test_Xs, test_ys, scalers = prepare_cv_test_data(scenarios, cv_df, test_df)
    
    ytrue = test_ys['Sc3']
    
    
    rmses_perfectfc = calc_scenario_rescalced_metrics(ytrue, metric=root_mean_squared_error,\
                                            preds_dict_path='data/results/test_perfect_wfc_preds.pkl',\
                                            scaler_dict_path='data/results/test_perfect_wfc_scalers.pkl')
    rmses_realfc = calc_scenario_rescalced_metrics(ytrue, metric=root_mean_squared_error,\
                                            preds_dict_path='data/results/test_real_wfc_preds.pkl',\
                                            scaler_dict_path='data/results/test_real_wfc_scalers.pkl')
    print rmses_perfectfc
    print rmses_realfc
    fig, axes = plt.subplots(2,1, sharey=True, figsize=(text_width, 0.8*text_width))
    
    titles = {'perfectfc':'(a) Perfect weather forecasts', 'realfc':'(b) Real weather forecasts'}
    scores = {'perfectfc':rmses_perfectfc, 'realfc':rmses_realfc}
    colors = {'OLS':'#1f77b4', 'MLP':'#ff7f0e', 'SVR':'#2ca02c'}#{'OLS':'#CC333F', 'MLP':'#EDC951', 'SVR':'#00A0B0'}
    offsets = {'OLS':-1, 'MLP':0, 'SVR':1}
    tick_labels = [sc_text[sc] for sc in scenarios]
    tick_pos = np.arange(len(tick_labels))      
            
    for ax, fc_type in zip(axes, ['perfectfc', 'realfc']):
        ax.set_axisbelow(True) # this sets the grid lines behind the bars
    
        for model in models:
            ax.bar(left=tick_pos+offsets[model]*bin_width, \
                   height=[scores[fc_type][sc][model] for sc in scenarios],
                   color=colors[model],
                   width=bin_width, label=model)
        
        ax.legend(ncol=3)
        ax.grid('on', linestyle=':')
        ax.set_ylabel('RMSE [MW]')
        ax.text(0.02, .9, titles[fc_type], transform=ax.transAxes, fontweight='bold', va='center')
        ax.xaxis.set_ticks(tick_pos)
        ax.xaxis.set_ticklabels(tick_labels)
        ax.set_ylim([0,50])
        
    fig.tight_layout()
    fig.savefig(savepath + 'perfect_real_fc_barplot.png', dpi=400)
    
    return fig
  
    
# figure 5
def err_vs_hour_per_month():
    cv_df = pd.read_pickle('data/cleaned/assembled_data/cv_data.pkl')
    test_df = pd.read_pickle('data/cleaned/assembled_data/test_data_real_fc.pkl')
    cv_Xs, cv_ys, test_Xs, test_ys, scalers = prepare_cv_test_data(scenarios, cv_df, test_df)
    
    ytrue = test_ys['Sc3']
    true_prod = pd.Series(scalers['Sc3'].inverse_transform_y(ytrue), index=test_df.index)
    
    with open('data/results/test_real_wfc_preds.pkl', 'r') as f:
        preds_dict = pickle.load(f)
    
    pred = preds_dict['Sc3']['SVR']
    pred_prod = pd.Series(scalers['Sc3'].inverse_transform_y(pred), index=test_df.index)
    
    metrics = {'RMSE':root_mean_squared_error, 'MAE':mean_absolute_error, 'MAPE':mean_absolute_percentage_error}
    errs = {}
    space_factor = 1
    for month in range(1,13):   
        errs[month] = {}
        month_true_prod = true_prod[true_prod.index.month==month]
        month_pred_prod = pred_prod[pred_prod.index.month==month]
        for metric in metrics.iterkeys():
            daily_errs = [metrics[metric](          \
                          month_true_prod[month_true_prod.index.hour==i], \
                          month_pred_prod[month_pred_prod.index.hour==i]\
                                          ) for i in range(24)]
            # This roll is done to ensure that hour 0 (between 23 and 00 comes last and hour 1 (00-01 comes first))
            daily_errs_rightts = np.roll(daily_errs, -1) 
            errs[month][metric] = daily_errs_rightts
            
    ocean_five = {'MAPE':'#CC333F', 'MAE':'#EDC951', 'RMSE':'#00A0B0'}        
    fig, axes = plt.subplots(5, 3, sharex=False, figsize=(text_width, 1.3*text_width), gridspec_kw={'height_ratios':[.15,1,1,1,1]})
    # legende axis is cax:
    cax = axes.flatten()[1]
    for a in axes.flatten()[0:3]:
        a.axis('off')
    
    axes = axes.flatten()[3:]
    fontsize=7
    hours = range(1,25)
    for month, ax in zip(range(1,13), axes):
        ax.set_title(calendar.month_name[month], fontsize=10, fontweight='bold')
        rmse_mae_lines = []
        for metric in ['RMSE', 'MAE']:
            prep_hours, prep_errs = prepend_for_axisspace(hours, errs[month][metric], space_factor)
            line, = ax.plot(prep_hours, prep_errs, color=ocean_five[metric], label=metric)
            rmse_mae_lines.append(line)
     
        metric = 'MAPE'
        second_ax = ax.twinx()
        app_hours, app_errs = append_for_axisspace(hours, 100*errs[month][metric], space_factor)
        mape_line, = second_ax.plot(app_hours, app_errs, c=ocean_five[metric], label=metric)
        second_ax.set_ylim(0,13.75)
        second_ax.set_ylabel('[%]', color=ocean_five['MAPE'], fontsize=fontsize)
        second_ax.tick_params('y', colors=ocean_five['MAPE'])
        second_ax.yaxis.set_tick_params(labelsize=fontsize)

        ax.set_xlim(1 - space_factor,24 + space_factor)
        ax.set_ylim(0,55)
        ax.set_ylabel('[MW]', fontsize=fontsize)
        ax.set_xlabel('Hour of day', fontsize=fontsize)
        ax.xaxis.set_tick_params(labelsize=fontsize)
    
        ax.set_xticks([1, 6, 12, 18, 24])
        ax.yaxis.set_tick_params(labelsize=fontsize)
        
        ax.grid('on', linestyle=':')
        fig.tight_layout(pad=0.5)
    
        leg = cax.legend(rmse_mae_lines + [mape_line], ['RMSE', 'MAE', 'MAPE'], \
                         ncol=3, fontsize=10, loc='upper center', bbox_to_anchor=(0.5, 0.85), title='Forecast error metric')
        
    fig.savefig(savepath + 'error_vs_hour_per_month.png', dpi=400)
    
    return fig


def prepend_for_axisspace(xvals, yvals, space_factor=0.5):
    new_xvals = [xvals[0] - space_factor*(xvals[1] - xvals[0])] + list(xvals)
    new_yvals = [yvals[0]] + list(yvals)
    
    return new_xvals, new_yvals


def append_for_axisspace(xvals, yvals, space_factor=0.5):
    new_xvals = list(xvals) + [xvals[-1] + space_factor*(xvals[-1] - xvals[-2])]
    new_yvals = list(yvals) + [yvals[-1]]
    
    return new_xvals, new_yvals


def err_dist_per_month():
    ocean_five = {'r':'#CC333F', 'y':'#EDC951', 'bl':'#00A0B0', 'br':'#6A4A3C', 'o':'#EB6841'}

    
    cv_df = pd.read_pickle('data/cleaned/assembled_data/cv_data.pkl')
    test_df = pd.read_pickle('data/cleaned/assembled_data/test_data_real_fc.pkl')
    cv_Xs, cv_ys, test_Xs, test_ys, scalers = prepare_cv_test_data(scenarios, cv_df, test_df)
    
    ytrue = test_ys['Sc3']
    true_prod = pd.Series(scalers['Sc3'].inverse_transform_y(ytrue), index=test_df.index)
    
    with open('data/results/test_real_wfc_preds.pkl', 'r') as f:
        preds_dict = pickle.load(f)
    
    pred = preds_dict['Sc3']['SVR']
    pred_prod = pd.Series(scalers['Sc3'].inverse_transform_y(pred), index=test_df.index)
    
    err_tss = {}
    for month in range(1,13):   
        month_true_prod = true_prod[true_prod.index.month==month]
        month_pred_prod = pred_prod[pred_prod.index.month==month]
        err_tss[month] = month_pred_prod - month_true_prod
            
    fig, axes = plt.subplots(5, 3, figsize=(text_width, 1.2*text_width), gridspec_kw={'height_ratios':[.1,1,1,1,1]})
    cax = axes.flatten()[1]
    for a in axes.flatten()[0:3]:
        a.axis('off')
    axes = axes.flatten()[3:]
    xmin = -175
    xmax = 175
    ymin = 0
    ymax = 170
    fontsize = 7
    for month, ax in zip(range(1,13), axes.flatten()):

        count, bins = np.histogram(err_tss[month], bins='scott')
        ax.hist(err_tss[month], bins=bins, ec='k', fc=ocean_five['bl'], lw=0.3, normed=KDE, label='Histogram')
        ax.set_title(calendar.month_name[month], fontsize=10, fontweight='bold')
        
        ax.grid('on', linestyle=':')
        
        q10 = np.percentile(err_tss[month], 10.)
        q90 = np.percentile(err_tss[month], 90)
        
        ax.vlines(q10, ymin, ymax, ocean_five['r'], label='10% quantile')
        ax.vlines(q90, ymin, ymax, ocean_five['o'], label='90% quantile')
        
        ax.set_xlabel('Forecast error [MW]', fontsize=fontsize)
        ax.set_ylabel('Number of hours', fontsize=fontsize)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.xaxis.set_ticks(np.linspace(xmin+25, xmax-25, 5))
        ax.xaxis.set_tick_params(labelsize=fontsize)
        ax.yaxis.set_tick_params(labelsize=fontsize)
        ax.set_axisbelow(True) # this sets the grid lines behind the bars

        print "%s \t Q10=%2.2f MW \t Q90=%2.2f MW" % (calendar.month_name[month], q10, q90)
    handles, labels = ax.get_legend_handles_labels()
    leg = cax.legend(handles, labels, ncol=3, fontsize=10, loc='upper center', bbox_to_anchor=(0.5, 0.5))

        
    fig.tight_layout()
    fig.savefig(savepath + 'err_dist_per_month.png', dpi=400)


def holiday_err_barplot():
    ocean_five = {'r':'#CC333F', 'y':'#EDC951', 'bl':'#00A0B0', 'br':'#6A4A3C', 'o':'#EB6841'}
    
    cv_df = pd.read_pickle('data/cleaned/assembled_data/cv_data.pkl')
    test_df = pd.read_pickle('data/cleaned/assembled_data/test_data_real_fc.pkl')
    test_df['any holiday'] = test_df['school_holiday'] | test_df['observance'] | test_df['national_holiday']
    cv_Xs, cv_ys, test_Xs, test_ys, scalers = prepare_cv_test_data(scenarios, cv_df, test_df)
    
    ytrue = test_ys['Sc3']
    true_prod = pd.Series(scalers['Sc3'].inverse_transform_y(ytrue), index=test_df.index)
    
    
  
    
    with open('data/results/test_real_wfc_preds.pkl', 'r') as f:
        preds_dict = pickle.load(f)
    
    pred_prod = {sc:pd.Series(scalers['Sc3'].inverse_transform_y(preds_dict[sc]['SVR']), index=test_df.index)\
                 for sc in scenarios}

    masks = OrderedDict({'All days':[True]*len(test_df),
                         'Weekdays':(~test_df['weekend']) & (~test_df['any holiday']),
                         'Weekends':test_df['weekend'] & (~test_df['any holiday']),
                         'Holidays':test_df['any holiday']})

    fig, axes = plt.subplots(2,1, figsize=(text_width, .67*text_width), gridspec_kw={'height_ratios':[0.07, 1]})
    binwidth = 0.2
    offsets = {sc:i for sc, i in zip(scenarios, range(-1,2))}
    colors = {'Sc1':ocean_five['r'], 'Sc2':ocean_five['y'], 'Sc3':ocean_five['bl']}
    xtickpos = np.arange(len(masks))
    
    err_dict = {}
    ax = axes[1]
    lax = axes[0]  
    lax.axis('off')

    for sc in scenarios:
        err_dict[sc] = []
        for daytype in masks.iterkeys():
            mask = masks[daytype]
            err = root_mean_squared_error(true_prod[mask], pred_prod[sc][mask])
            err_dict[sc].append(err)
            print daytype, sum(masks[daytype])/24.
            
        ax.bar(left=xtickpos + offsets[sc]*binwidth, height=err_dict[sc], fc=colors[sc], width=binwidth, label=sc_text[sc])
    ax.xaxis.set_ticks(xtickpos)
    ax.xaxis.set_ticklabels(masks.keys())
    ax.set_ylabel('RMSE [MW]')
    
    lax.legend(*ax.get_legend_handles_labels(), ncol=3, loc='center', bbox_to_anchor=(0.49, 0.35))
    ax.grid('on', linestyle=':')
    ax.set_axisbelow(True) # this sets the grid lines behind the bars
    fig.tight_layout(pad=0.5)
    fig.savefig(savepath + 'holiday_err_barplot.png', dpi=400)
    
    return test_df


def create_month_err_table(save=False):
    cv_df = pd.read_pickle('data/cleaned/assembled_data/cv_data.pkl')
    test_df = pd.read_pickle('data/cleaned/assembled_data/test_data_real_fc.pkl')
    cv_Xs, cv_ys, test_Xs, test_ys, scalers = prepare_cv_test_data(scenarios, cv_df, test_df)
    
    ytrue = test_ys['Sc3']
    true_prod = pd.Series(scalers['Sc3'].inverse_transform_y(ytrue), index=test_df.index)
    
    with open('data/results/test_real_wfc_preds.pkl', 'r') as f:
        preds_dict = pickle.load(f)
    
    pred = preds_dict['Sc3']['SVR']
    pred_prod = pd.Series(scalers['Sc3'].inverse_transform_y(pred), index=test_df.index)
    
    df = pd.DataFrame(columns=['RMSE', 'ME', 'Q10', 'Q90', 'Q1', 'Q99', 'nQ10', 'nQ90', 'nQ1', 'nQ99',], index=[calendar.month_name[i] for i in range(1,13)])
    for month in range(1,13):   
        month_true_prod = true_prod[true_prod.index.month==month]
        month_pred_prod = pred_prod[pred_prod.index.month==month]
        
        mix = calendar.month_name[month]
        df.at[mix, 'RMSE'] = root_mean_squared_error(month_true_prod, month_pred_prod)
        df.at[mix, 'ME'] = mean_error(month_true_prod, month_pred_prod)
        err = month_pred_prod - month_true_prod
        for q in [10, 90, 1, 99]:
            df.at[mix, 'Q%i'%q] = np.percentile(err, q)
            df.at[mix, 'nQ%i'%q] = np.percentile(err, q)/np.mean(month_true_prod)

    print df    
    if save:
        df.to_excel('data/monthly_error_table.xls')
    
    return df