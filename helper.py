# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 10:28:12 2024

@author: Anssi Laukkarinen

Python module file in the "csv_to_epw" github repository.

"""

import numpy as np





def get_epw_timestamps(df_):
    
    # The input data has hours 0-23
    # We have to adjust timestamps where hour is 0
    # Other timestamps can be kept as they are
    
    # Both are already defined in such a way that the timestamps represents
    # the previous hour
    
    year_output = 2023
    
    timestamps = []
    
    for idx, row in df_.iterrows():
        
        year = 999
        month = 999
        day = 999
        hour = 999
        
        if idx.hour == 0:
            # First hour of the day
            
            
            if idx.day == 1:
                # First day of month
                # The length of the previous month varies
                
                if idx.month in [1]:
                    # First month of the year
                    # Year changes also
                    
                    # TODO: The current change of year doesn't work universally
                    # It assumes that there is no timestamp for the yyyy-01-01T00:00,
                    # but the last row is duplicate of the second-last row.
                    hour = 24
                    day = 31
                    month = 12
                    year = year_output
                    
                elif idx.month in [3]:
                    # 0-23 system: 1.3.2023 00:00:00 == 1-24 system: 28.2.2023 24:00:00,
                    # when there is no leap days. Year 2023 doesn't have leap days.
                    hour = 24
                    day = 28
                    month = idx.month - 1
                    year = year_output
                
                # Other months of the year
                elif idx.month in [5, 7, 10, 12]:
                    hour = 24
                    day = 30
                    month = idx.month - 1
                    year = year_output
                    
                elif idx.month in [2, 4, 6, 8, 9, 11]:
                    hour = 24
                    day = 31
                    month = idx.month - 1
                    year = year_output
                    
                else:
                    print('unknown month, error')
                
            else:
                # Other days of the month
                hour = 24
                day = idx.day - 1
                month = idx.month
                year = year_output
                
            
            
            
        else:
            # Other hours of the day
            
            year = year_output
            month = idx.month
            day = idx.day
            hour = idx.hour
        
        timestamp = [year, month, day, hour, 0, 0]
        timestamps.append(timestamp)
    
    
    timestamps_np = np.array(timestamps)
    
    return(timestamps_np)




def calc_pvsat_water(Tdb_):
    
    # Saturation vapor pressure is calculated alwasy with regards 
    # to liquid water
    
    pvsat = 610.5 * np.exp((17.269*Tdb_)/(237.3+Tdb_))
    
    return(pvsat)




def calc_Tdp(Tdb_, RHwater_):
    
    # [Tdp] = degC
    
    pv = (RHwater_/100.0) * calc_pvsat_water(Tdb_)
    
    Tdp_ = []
    
    pvval_previous = 500.0
    
    for pvval in pv:
        # print(pvval, flush=True)
        
        if pvval < 10.0:
            pvval = pvval_previous
            pvval_previous = 500.0
        else:
            pvval_previous = pvval
        
        if pvval >= 610.5:
            
            theta = (237.3*np.log(pvval/610.5)) / (17.269-np.log(pvval/610.5))
        
        else:
            
            theta = (265.5*np.log(pvval/610.5)) / (21.875-np.log(pvval/610.5))
        
        
        Tdp_.append(theta)
    
    Tdp_ = np.array(Tdp_)    
    
    return(Tdp_)





def calc_LWdn(input_dict, method='dTsky'):
    
    
    sigma_SB = 5.67e-8    
    
    if method == 'dTsky':
        
        Tdb_ = input_dict['Tdb']
        
        dT_sky = 11.0
        Tsky_K = (Tdb_+273.15) - dT_sky
        LWdn = sigma_SB * Tsky_K**4
        
        return(LWdn)
    
    elif method == 'clearness':
        
        T2m_K = input_dict['Tdb'] + 273.15
        phi_ = input_dict['RH'] / 100.0
        
        pvsat_ = calc_pvsat_water(input_dict['Tdb'])
        e_a_in_hPa = phi_ * pvsat_ / 100.0
        
        ghi_extra_ = input_dict['ghi_extra']    
        ghi_ = input_dict['ghi']
        
        # Clearness index Kt, 0...1
        Kt = ghi_ / ghi_extra_
        idxs = ghi_ < 20.0
        Kt.iloc[idxs] = np.nan
        Kt.interpolate(inplace=True)
        
        idxs_nans = Kt.isna()
        
        if np.sum(idxs_nans) > 0:
            print('  Kt n_nans:', np.sum(idxs_nans), flush=True)
            Kt.iloc[idxs_nans] = 0.5
        
        
        # Cloudiness c, 0...1
        # x = Kt.values
        # xp = (0.25, 0.8)
        # fp = (1.0, 0.0)
        # c = np.interp(x, xp, fp)
        
        # fig, ax = plt.subplots()
        # ax.plot(xp, fp, label='Flertchinger')
        # ax.plot((0,1), (1, 0), label='linear')
        # ax.legend()
        
        # cloud fraction f_c
        f_c = 1.0 - Kt.values
        f_c = np.maximum(f_c, 0.0)
        f_c = np.minimum(f_c, 1.0)
            
        
        # emissivitites
        emiss_clear_sky = 1.24 * (e_a_in_hPa/T2m_K)**(1/7)
        emiss_cloudy_sky = 1.0
        
        emiss_all_sky = f_c * emiss_cloudy_sky + (1-f_c) * emiss_clear_sky
        
        LWdn = emiss_all_sky * sigma_SB * T2m_K**4
        
        return(LWdn)
    
        




def calc_N_from_LWdn(LWdn_, Tdb_, Tdp_):
    
    sigma_SB = 5.67e-8
    
    # If LWdn is missing, it is calculated from opaque sky cover
    # aN**3 + bN**2 + cN + d = 0 -> to find N, solve roots of polynomial
    x = 0.787 + 0.764*np.log((Tdp_+273.15)/273.15)
    
    ds = 1 - LWdn_ / (x * sigma_SB * (Tdb_+273.15)**4)
    
    c_coef = 0.0224
    b_coef = 0.0035
    a_coef = 0.00028
    
    N_list = []
    
    for d_coef in ds:
        
        coefs = [a_coef, b_coef, c_coef, d_coef]
        
        roots_from_np = np.roots(coefs)
        idxs_real = np.isreal(roots_from_np)
        
        if idxs_real.sum() == 0:
            # no solution
            
            N = 5.0
            
        elif idxs_real.sum() == 1:
            # one real solution
            
            N = np.real(roots_from_np[idxs_real])
            
        elif idxs_real.sum() > 1:
            # more than one real solution
            
            Ns = np.real(roots_from_np[idxs_real])
            
            idxs_inRange = (Ns >= 0.0) & (Ns <= 10.0)
            
            if idxs_inRange.sum() == 0:
                # No solutions in range
                
                N = 5.0
                
            elif idxs_inRange.sum() == 1:
                # One solution in range
                
                N = Ns[idxs_inRange]
                
            elif idxs_inRange.sum() > 1:
                # More than one solution in range
                # Use the average
                Ns_inRange = Ns[idxs_inRange]
                N = np.mean(Ns_inRange)
        
        N_list.append(N)
    
    N_np = np.array(N_list).flatten()
    
    N_np = np.maximum(N_np, 0.0)
    N_np = np.minimum(N_np, 10.0)
    
    return(N_np)


