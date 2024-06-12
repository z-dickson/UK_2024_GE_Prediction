import pandas as pd
import numpy as np
import polars as pl
import os
import pyreadstat 
from tqdm import tqdm


# import geopandas to plot maps 
import geopandas as gpd
from shapely.geometry import Point

#dpath = '/Users/z.dickson/Library/CloudStorage/Dropbox-DeptofMethodology/Zachary Dickson/main/dickson_main/2024_UK_GE/data/'

def get_data(url):
    if url.endswith('csv'):
        return pd.read_csv(url)
    elif url.endswith('xlsx'):\
        return pd.read_excel(url)
    else:
        return None
    
    
    
def yougov_polls(sheet_name):
    try: 
        df = pd.read_excel('data/yougov_polling_regional.xlsx', sheet_name=sheet_name)
    except FileNotFoundError:
        print("""File not found - choose 'All adults', 'London', 'Rest of South', 'Midlands', 'North', 'Scotland', 'Wales', 'Male','Female', '18-24', 25-49', '50-64', '65+', 'Conservative', 'Labour', 'Liberal Democrat','Remain', 'Leave', 'ABC1', 'C2DE'""")
    #    return None
    df=df.transpose()
    df.columns = df.iloc[0]
    df = df[1:]
    df = df.reset_index()
    df.rename(columns={'index':'date'}, inplace=True)
    df['population'] = sheet_name
    
    return df 
    
    
    
    
    
def get_BES_data(df, min_vars = 20, max_vars = 25):

    #df = pd.read_stata(path)

    # get a list of columns that end with 'WXX' 

    cols = [c for c in df.columns if c.endswith('W25') or c.endswith('W24') or c.endswith('W23') or c.endswith('W22') or c.endswith('W21') or c.endswith('W20') or c.endswith('W19') or c.endswith('W18') or c.endswith('W17') or c.endswith('W16') or c.endswith('W15') or c.endswith('W14') or c.endswith('W13') or c.endswith('W12') or c.endswith('W11') or c.endswith('W10') or c.endswith('W9') or c.endswith('W8') or c.endswith('W7') or c.endswith('W6') or c.endswith('W5') or c.endswith('W4') or c.endswith('W3') or c.endswith('W2') or c.endswith('W1')]

    cols_df = pd.DataFrame(cols, columns=['col_name'])
    cols_df['var_name'] = cols_df.col_name.str.split('W').str[0]
    x = cols_df.groupby('var_name').count().sort_values('col_name', ascending=False).reset_index()

    cols_to_keep = []
    for i in x.loc[(x.col_name >= min_vars) & (x.col_name <= max_vars)].var_name.values:
        for j in cols_df.loc[cols_df['var_name']==i].col_name.unique():
            cols_to_keep.append(j)
    #return cols_to_keep
    df = df[cols_to_keep + ['id']]
    df = pd.melt(df, id_vars='id', value_vars=cols_to_keep)
    df['wave'] = df.variable.str.slice(-2, )
    df.wave = df.wave.str.replace('W', '').astype(int)
    df['variable_name'] = df.variable.str.slice(0, -2)
    df.variable_name = df.variable_name.str.replace('W', '')
    # drop all the weight variables which adds like 8 million rows 
    #df = df.loc[~df.variable.str.contains('wt_new')]
    df = df.pivot_table(index=['id', 'wave'], columns='variable_name', values='value', aggfunc='first').reset_index()
    return df








def merge_results(df, result, election_year):
       election_year_str = str(election_year)[2:]
       con_dict = dict(zip(result['ons_const_id'], result['con_' + election_year_str]))
       lab_dict = dict(zip(result['ons_const_id'], result['lab_' + election_year_str]))
       #x = pd.merge(results[['ons_const_id', 'winner_' + year, 'con_' + year, 'lab_' + year, 'snp_' + year, 'pc_' + year, 'ukip_' + year,
       #                      'green_' + year, 'other_' + year, 'turnout_' + year]], df, left_on = 'ons_const_id', right_on = 'pcon_code_imputed', how = 'right')
       df.loc[df.election == election_year, 'con_target'] = df.loc[df.election == election_year, 'pcon_code_imputed'].map(con_dict)
       
       df.loc[df.election == election_year, 'lab_target'] = df.loc[df.election == election_year, 'pcon_code_imputed'].map(lab_dict)
       


       return df
   
   
   ## convert age to numeric
def convert_age(df):
    df.loc[df.age == '85+', 'age'] = 86
    df.loc[df.age == 'Under 18', 'age'] = 17
    df.age = df.age.astype(float)
    return df


# drop rows with no constituency data
def drop_dups(df, target_var):
    df = df.loc[~df[target_var].isna()]
    return df




# impute MSOA codes 



def impute_loc_codes(df):
    df.loc[df.msoa11 == '', 'msoa11'] = np.nan
    df.loc[df.pcon_code == '', 'pcon_code'] = np.nan
    
    pcon_dict = df[['id','pcon_code']].dropna()
    pcon_dict = pcon_dict.drop_duplicates()
    pcon_dict = dict(zip(pcon_dict['id'], pcon_dict['pcon_code']))

    df['pcon_code_imputed'] = df['id'].map(pcon_dict)

    msoa_dict = df[['id','msoa11']].dropna()
    msoa_dict = msoa_dict.drop_duplicates()
    msoa_dict = dict(zip(msoa_dict['id'], msoa_dict['msoa11']))

    df['msoa11_imputed'] = df['id'].map(msoa_dict)
    return df 
    
    
    
# vars to convert to numeric 


def create_dummies(df):
    vars_to_convert = ['leftRight', 'likeCon', 'likeLD', 'likeLab',
        'likePC', 'likeSNP', 'p_hh_children', 'p_hh_size', 'age']

    vars_to_ignore = ['starttime', 'endtime', 'id', 'wave', 'pcon_code', 'msoa11', 'pcon_code_imputed', 'msoa11_imputed', 'oslaua', 'oslaua_code', 'pcon', 'pano']

    for var in vars_to_convert:
        df[var] = pd.to_numeric(df[var], errors='coerce')
        
    for var in tqdm(df.columns): 
        if var not in vars_to_convert and var not in vars_to_ignore:
            df = pd.concat([df, pd.get_dummies(df[var], prefix=var)], axis = 1)
            
    return df


def assign_elections(df):
    
    df.loc[(df.wave > 2) & (df.wave <= 5), 'election'] = 2015
    df.loc[(df.wave > 9) & (df.wave <= 12), 'election'] = 2017
    df.loc[(df.wave > 13) & (df.wave < 17), 'election'] = 2019
    df.loc[df.wave >= 23, 'election'] = 2024

    df = df.loc[df.election.notnull()]
    return df 


def add_missing_ons_vars(old_df, ndf): 
    
# map msoa11_imputed to pcon_code_imputed

    msoa_dict = old_df[['msoa11_imputed','pcon_code_imputed']].dropna()
    msoa_dict = msoa_dict.drop_duplicates()
    msoa_dict = dict(zip(msoa_dict['pcon_code_imputed'], msoa_dict['msoa11_imputed']))
    ndf['msoa11_imputed'] = ndf['pcon_code_imputed'].map(msoa_dict)
    
    return ndf 



def read_dem_data(path, df):
    x = pd.read_csv(path)
    try: 
        df = pd.merge(df, x, left_on='PCON25CD_imputed', right_on = 'ons_code', how='left')
    except KeyError:
        df = pd.merge(df, x, left_on='PCON25CD_imputed', right_on = 'gss_code', how='left')
    return df 



# Reporting util for different optimizers
def report_perf(optimizer, X, y, title="model", callbacks=None):
    """
    A wrapper for measuring time and performances of different optmizers
    
    optimizer = a sklearn or a skopt optimizer
    X = the training set 
    y = our target
    title = a string label for the experiment
    """
    start = time()
    
    if callbacks is not None:
        optimizer.fit(X, y, callback=callbacks)
    else:
        optimizer.fit(X, y)
        
    d=pd.DataFrame(optimizer.cv_results_)
    best_score = optimizer.best_score_
    best_score_std = d.iloc[optimizer.best_index_].std_test_score
    best_params = optimizer.best_params_
    
    print((title + " took %.2f seconds,  candidates checked: %d, best CV score: %.3f "
           + u"\u00B1"+" %.3f") % (time() - start, 
                                   len(optimizer.cv_results_['params']),
                                   best_score,
                                   best_score_std))    
    print('Best parameters:')
    print(best_params)
    print()
    return best_params