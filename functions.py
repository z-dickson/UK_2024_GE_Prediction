import pandas as pd
import numpy as np
import polars as pl
import os
import pyreadstat 
from tqdm import tqdm
from time import time
import statsmodels.api as sm
# create xgboost prediction model to predict the 2015 election results 

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer

# Skopt functions
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, DeltaYStopper
from skopt.space import Real, Categorical, Integer


import pprint
import joblib
from functools import partial

# Data processing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Model selection
from sklearn.model_selection import KFold, StratifiedKFold

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
    
    
def index_by_constituency(df, col, new_col_name):
    x = df.groupby(['Westminster Parliamentary constituencies Code', col]).agg({'Observation': 'sum'}).reset_index()
    x[new_col_name] = x['Observation'] / x.groupby(['Westminster Parliamentary constituencies Code'])['Observation'].transform('sum')
    ## Merge the data so that we have new columns for each of the categories
    x = x.pivot(index='Westminster Parliamentary constituencies Code', columns=col).reset_index()
    return x

    
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
    df.rename(columns={'index':'date','If there were a general election held tomorrow, which party would you vote for?': 'poll'}, inplace=True)
    
    df['demographic'] = sheet_name
    # add suffix to columns 
    #df.columns = [f'vote_intention_{c}' for c in df.columns]
    
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



def get_BES_data_v2(df, min_vars=20, max_vars=25):
    # Efficiently filter columns that end with the specified wave suffixes
    suffixes = [f'W{i}' for i in range(1, 26)]
    cols = [c for c in df.columns if any(c.endswith(suffix) for suffix in suffixes)]
    
    # Extract variable names without using an intermediate DataFrame
    var_name_counts = {}
    for col in cols:
        var_name = col.split('W')[0]
        if var_name in var_name_counts:
            var_name_counts[var_name] += 1
        else:
            var_name_counts[var_name] = 1
    
    # Select variable names that meet the min_vars and max_vars criteria
    cols_to_keep = [col for col in cols if min_vars <= var_name_counts[col.split('W')[0]] <= max_vars]
    
    # Filter the DataFrame to keep only the necessary columns
    df = df[cols_to_keep + ['id']]
    
    # Reshape the DataFrame using melt and pivot_table
    df_melted = pd.melt(df, id_vars='id', value_vars=cols_to_keep)
    df_melted['wave'] = df_melted['variable'].str.extract(r'W(\d+)$').astype(int)
    df_melted['variable_name'] = df_melted['variable'].str.replace(r'W\d+$', '', regex=True)
    
    df_pivoted = df_melted.pivot_table(index=['id', 'wave'], columns='variable_name', values='value', aggfunc='first').reset_index()
    
    return df_pivoted




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


def create_dummies(df, cols):
    for col in tqdm(cols):
        df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis = 1)
    return df


def create_factors(df, cols):
    for col in cols:
        df[f'{col}_factor'] = df[col].factorize()[0]
    return df


def assign_elections(df):
    
    df.loc[(df.wave > 4) & (df.wave <= 5), 'election'] = 2015
    df.loc[(df.wave > 11) & (df.wave <= 12), 'election'] = 2017
    df.loc[(df.wave > 16) & (df.wave <= 17), 'election'] = 2019
    df.loc[df.wave >= 25, 'election'] = 2024

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




def xgboost_model(cols_to_drop,
                  df,
                  target_var,
                  search_space,
                  time, 
                  folds):

    X = df.drop(cols_to_drop, axis = 1)

    Y = df[target_var]

    #from sklearn.preprocessing import StandardScaler
    #scaler = StandardScaler()


    #x_Scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)

    X_train = X_train.set_index(['PCON25CD_imputed', 'election'])
    X_test = X_test.set_index(['PCON25CD_imputed', 'election'])


    y_stratified = pd.cut(y_train.rank(method='first'), bins=10, labels=False)

    # Winsorizing lower bounds
    from scipy.stats.mstats import winsorize
    y_train = np.array(winsorize(y_train, [0.008, 0.0]))
    
    # Reporting util for different optimizers

    # Setting the scoring function
    scoring = make_scorer(partial(mean_squared_error, squared=False), 
                        greater_is_better=False)

    # Setting the validation strategy
    skf = StratifiedKFold(n_splits=7,
                        shuffle=True, 
                        random_state=0)

    cv_strategy = list(skf.split(X_train, y_stratified))
    
    # Wrapping everything up into the Bayesian optimizer
    opt = BayesSearchCV(estimator=reg,                                    
                        search_spaces=search_spaces,                      
                        scoring=scoring,                                  
                        cv=cv_strategy,                                           
                        n_iter=120,                                       # max number of trials
                        n_points=1,                                       # number of hyperparameter sets evaluated at the same time
                        n_jobs=1,                                         # number of jobs
                        iid=False,                                        # if not iid it optimizes on the cv score
                        return_train_score=False,                         
                        refit=False,                                      
                        optimizer_kwargs={'base_estimator': 'GP'},        # optmizer parameters: we use Gaussian Process (GP)
                        random_state=0)  
    
    from time import time

    # Running the optimizer
    overdone_control = DeltaYStopper(delta=0.0001)                    # We stop if the gain of the optimization becomes too small
    time_limit_control = DeadlineStopper(total_time=time)          # We impose a time limit (1000 seconds)
    
    
    best_params = report_perf(opt, X_train, y_train,'XGBoost_regression', 
                            callbacks=[overdone_control, time_limit_control])
    
    # Transferring the best parameters to our basic regressor
    reg = xgb.XGBRegressor(random_state=0, booster='gbtree', objective='reg:squarederror', tree_method='gpu_hist', **best_params)
    
    
    # Cross-validation prediction
    folds = folds
    skf = StratifiedKFold(n_splits=folds,
                        shuffle=True, 
                        random_state=0)

    predictions = np.zeros(len(X_test))
    rmse = list()

    for k, (train_idx, val_idx) in enumerate(skf.split(X_train, y_stratified)):
        reg.fit(X_train.iloc[train_idx, :], y_train[train_idx])
        val_preds = reg.predict(X_train.iloc[val_idx, :])
        val_rmse = mean_squared_error(y_true=y_train[val_idx], y_pred=val_preds, squared=False)
        print(f"Fold {k} RMSE: {val_rmse:0.5f}")
        rmse.append(val_rmse)
        predictions += reg.predict(X_test).ravel()
        
    predictions /= folds
    print(f"repeated CV RMSE: {np.mean(rmse):0.5f} (std={np.std(rmse):0.5f})")
    
    print(str(best_params))
    # Preparing the submission
    submission = pd.DataFrame({'id':X_test.index, 
                            'target': predictions})
    
    return reg





# Function to harmonize census data for joint probability estimation

def harmonize_census_data(df):
    df.loc[df['Highest level of qualification (7 categories) Code'] == 4, 'social_grade'] = 'ABC1'
    df.loc[df['Highest level of qualification (7 categories) Code'] != 4, 'social_grade'] = 'C2DE'
    df['gender'] = df['Sex (2 categories)']
    df.loc[df['Age (6 categories) Code'] == 2, 'age'] = '18-24'
    df.loc[df['Age (6 categories) Code'] == 3, 'age'] = '25-49'
    df.loc[df['Age (6 categories) Code'] == 4, 'age'] = '25-49'
    df.loc[df['Age (6 categories) Code'] == 5, 'age'] = '50-64'
    df.loc[df['Age (6 categories) Code'] == 6, 'age'] = '65+'
    df['pcon_code'] = df['Westminster Parliamentary constituencies Code']
    df = df.groupby(['pcon_code', 
                'social_grade',
                'age', 
                'gender']).agg({'Observation': 'sum'}).reset_index()
    
    # create a total column for the total number of people in each constituency
    df['total_population'] = df.groupby('pcon_code')['Observation'].transform('sum')
    return df 





def create_cd_data_agg(df):

    cols_to_index = ['Highest level of qualification (7 categories)', 'Sex (2 categories)', 'Household deprivation (6 categories)', 'Age (6 categories)']


    con_df = pd.DataFrame()
    for col in cols_to_index:
        z = index_by_constituency(df, col, col + '_comp')
        con_df = pd.concat([con_df, z], axis=1)
        
        
    # drop the multi-level columns
    con_df.columns = [' '.join(col).strip() for col in con_df.columns.values]


    abc1 = ['Highest level of qualification (7 categories)_comp Level 4 qualifications or above: degree (BA, BSc), higher degree (MA, PhD, PGCE), NVQ level 4 to 5, HNC, HND, RSA Higher Diploma, BTEC Higher level, professional qualifications (for example, teaching, nursing, accountancy)']

    cde2 = ['Highest level of qualification (7 categories)_comp Does not apply', 
            'Highest level of qualification (7 categories)_comp Level 1 and entry level qualifications: 1 to 4 GCSEs grade A* to C, Any GCSEs at other grades, O levels or CSEs (any grades), 1 AS level, NVQ level 1, Foundation GNVQ, Basic or Essential Skills',
            'Highest level of qualification (7 categories)_comp Level 2 qualifications: 5 or more GCSEs (A* to C or 9 to 4), O levels (passes), CSEs (grade 1), School Certification, 1 A level, 2 to 3 AS levels, VCEs, Intermediate or Higher Diploma, Welsh Baccalaureate Intermediate Diploma, NVQ level 2, Intermediate GNVQ, City and Guilds Craft, BTEC First or General Diploma, RSA Diploma',
        'Highest level of qualification (7 categories)_comp Level 3 qualifications: 2 or more A levels or VCEs, 4 or more AS levels, Higher School Certificate, Progression or Advanced Diploma, Welsh Baccalaureate Advance Diploma, NVQ level 3; Advanced GNVQ, City and Guilds Advanced Craft, ONC, OND, BTEC National, RSA Advanced Diploma',
        'Highest level of qualification (7 categories)_comp No qualifications',
        'Highest level of qualification (7 categories)_comp Other: apprenticeships, vocational or work-related qualifications, other qualifications achieved in England or Wales, qualifications achieved outside England or Wales (equivalent not stated or unknown)']

       
    con_df.rename(columns={'Age (6 categories)_comp Aged 15 years and under': 'age_15',
            'Age (6 categories)_comp Aged 16 to 24 years': '18_24',
            'Age (6 categories)_comp Aged 25 to 34 years': '25_34',
            'Age (6 categories)_comp Aged 35 to 49 years': '35_49',
            'Age (6 categories)_comp Aged 50 to 64 years': '50_64',
            'Age (6 categories)_comp Aged 65 years and over': '65+',
            'Sex (2 categories)_comp Female': 'Female', 
            'Sex (2 categories)_comp Male': 'Male',
            'abc1': 'ABC1', 
            'cde2': 'C2DE',
            'Westminster Parliamentary constituencies Code': 'pcon_code'}, inplace=True)

    con_df['25_49'] = con_df['25_34'] + con_df['35_49']

    # create a new column for the percentage of people with a degree
    con_df['abc1'] = con_df[abc1].sum(axis=1)
    con_df['cde2'] = con_df[cde2].sum(axis=1)
    
    return con_df
