import numpy  as np
import pandas as pd
import numba

numba.jit()
def dectection_rate(n:int):
    
    """ Estimates the time that has passed for an individual to be diagnosed."""
    return np.random.normal(0, 1, (1,n))

numba.jit()
def moderate_recovery(n:int):
    """ Estimates the time of recovery for a moderate case."""
    
    return np.random.normal(9.5, 2.5/2, (1,n))

numba.jit()
def severe_recovery(n:int):
    """ Estimates the time of recovery for a severe case."""
    
    recovery = np.random.normal(14, 7/2, (1,n))
    recovery[np.where(recovery < 7)] = 7
    return recovery

numba.jit()
def estimate_recovery(I:int):
    """ Estimates the total time of recovery for I new infections."""
    
    # Split the infections into moderate, severe
    m, s, _ = np.round(np.array([0.804, 0.176,0.018]) * I).astype(int)
    
    recovery = dectection_rate(m + s)
    recovery = np.round(recovery + 
                        np.hstack((moderate_recovery(m), severe_recovery(s)))).astype(int)
    
    return recovery

def approx_recoveries(infections:np.ndarray):
    """ Estimates the time of recovery for a sequence of infections. """
    
    daily_recoveries = np.zeros(infections.shape[0])
    for day, daily_infections in enumerate(infections):
        days_till_recovery, recoveries = np.unique(estimate_recovery(daily_infections), 
                                                   return_counts = True)
        day_of_recovery = day + days_till_recovery
        try:
            daily_recoveries[day_of_recovery] = daily_recoveries[day_of_recovery] + recoveries
        except:
            pass
    
    return daily_recoveries

def split(df, start:any, end:any, column_name:str):
    """ Split a pandas DataFrame a"""
    start_row_index = df.index[df.loc[:, column_name] == start][0]
    end_row_index   = df.index[df.loc[:, column_name] == end][0]
    return df.iloc[end_row_index:start_row_index, :]