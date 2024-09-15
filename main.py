# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 22:16:51 2024

@author: Anssi Laukkarinen


Input file column names:
STEP;YEAR;MON;DAY;HOUR;TEMP;RH;WS;WDIR;GHI;DHI;DNI



"""

import os
import numpy as np
import pandas as pd

import pvlib


root_folder = r'C:\Users\laukkara\github\txt_to_epw'

input_folder = os.path.join(root_folder,
                            'input')

output_folder = os.path.join(root_folder,
                             'output')

if not os.path.exists(output_folder):
    os.makedirs(output_folder)




###################

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












###################

data = {}


for file in os.listdir(input_folder):
    
    print(file)
    
    # Read in file
    fname = os.path.join(input_folder,
                         file)
    
    df = pd.read_csv(fname,
                     sep=';',
                     index_col=False,
                     comment='#')

    
    # Remove leap days
    idxs_to_drop = (df.loc[:,'MON'] == 2) \
                    & (df.loc[:, 'DAY'] == 29)
    
    df = df.loc[~idxs_to_drop, :].copy()
    
    
    # Drop unneeded columns
    cols_to_drop = ['STEP','YEAR','MON', 'DAY', 'HOUR']
    
    df.drop(columns=cols_to_drop,
            inplace=True)  
    
    
    
    # 
    key = file[:-4].replace('-','_')
    data[key] = df


    ################
    
    
    df = pd.concat([df.iloc[1:,:], df.iloc[-1:,:]])
    df.reset_index(drop=True,
                   inplace=True)
    
    timestamps_pd = pd.date_range(start='2023-01-01 01:00:00',
                               end='2024-01-01 00:00:00',
                               freq='1h')
    
    df.index = timestamps_pd
    
    timestamps_np = get_epw_timestamps(df)
    
    n_hours = df.shape[0]
    
    
    location = file.split('_')[0].lower()
    
    if 'sod' in location:
        LAT = 67.37
        LON = 26.63
        ELEV = 179.0
        WMO_code = 7501
    
    elif 'jyv' in location:
        LAT = 62.4
        LON = 25.67
        ELEV = 139.0
        WMO_code = 2935
    
    elif 'jok' in location:
        LAT = 60.81
        LON = 23.5
        ELEV = 104.0
        WMO_code = 2963
    
    elif 'van' in location:
        LAT = 60.33
        LON = 24.97
        ELEV = 47.0
        WMO_code = 2974
    
    else:
        LAT = 60.81
        LON = 23.5
        ELEV = 104
        WMO_code = 2963
    
    
    
    header_rows = [f'LOCATION,{location},FIN,Finland,Finnish Meteorological Institute,{WMO_code},{LAT},{LON},2.0,{ELEV}',
                     'DESIGN CONDITIONS,0',
                     'TYPICAL/EXTREME PERIODS,0',
                     'GROUND TEMPERATURES,0',
                     'HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0',
                    f'COMMENTS 1,{file}',
                     'COMMENTS 2,"Finnish Test Reference Year by Finnish Meteorological Institute"',
                     'DATA PERIODS,1,1,Data,Monday,1/1,12/31']
    


    
    
    d = {}
    
    
    d['N1_year'] = timestamps_np[:,0]
    
    d['N2_month'] = timestamps_np[:,1]
    
    d['N3_day'] = timestamps_np[:,2]
    
    d['N4_hour'] = timestamps_np[:,3]
    
    d['N5_minute'] = timestamps_np[:,4]
    
    # All columns divided into groups: 4 2 4; 4 6 2 4 2
    # Uncertainty flag groups that are used in epw files
    flags1 = '?9?9?9?9'
    flags2 = 'E9?9D9?9'
    flags3 = '?9?9?9?9' # illuminance
    flags4 = '?9?9?9?9?9?9' # wind and clouds
    flags5 = '?9?9?9?9' # rain

    d['A1_flags'] = [flags1 + flags2 + flags3 + flags4 + flags5] * n_hours
    
    
    # 4 - meteorological
    d['N6_Tdb'] = df.loc[:,'TEMP'].values
    
    d['N7_Tdp'] = calc_Tdp(df.loc[:,'TEMP'].values, 
                           df.loc[:,'RH'].values)
    
    d['N8_RH'] = df.loc[:,'RH'].values
    
    d['N9_Patm'] = 101325.0*np.ones(n_hours)
    
    
    ## 2 - extraterrestrial radiation for calculating clearness index
    # Calculate extraterrestrial beam radiation
    dni_extra = pvlib.irradiance.get_extra_radiation( \
                    datetime_or_doy=df.index.shift(freq="-30T"))
    
    
    
    # Calculate extraterrestrial horizontal radiation
    # Assumed diffuse component is zero at top of atmosphere
    solpos = pvlib.solarposition.get_solarposition(
        time=df.index.shift(freq="-30min"),
        latitude=LAT,
        longitude=LON,
        altitude=ELEV,
        pressure=None,
        temperature=df.loc[:,'TEMP'].values)
    solpos.index = df.index  # reset index
    
    component_sum_df_extra = pvlib.irradiance.complete_irradiance(
                                        solar_zenith=solpos.zenith,
                                        dhi=0.0,
                                        dni=dni_extra.values)
    
    ghi_extra = component_sum_df_extra.loc[:,'ghi'].copy()
    ghi_extra.iloc[:] = np.maximum(ghi_extra, 0.0)
    
    d['N10_ExtraterrestrialHorizontalRadiation'] = ghi_extra.values # 9999 * np.ones(n_hours)
    
    d['N11_ExtraterrestrialDirectNormalRadiation'] = dni_extra.values # 9999 * np.ones(n_hours)
    
    
    ## Global radiation
    
    # dhi_zero = df.loc[:,'DHI'].copy()
    # dhi_zero.iloc[:] = 0.0
    
    ghi = df.loc[:,'GHI']
    
    
    ## Atmospheric downward long-wave radiation
    
    # fig, ax = plt.subplots()
    # ghi_extra.reset_index(drop=True).iloc[3000:3100].plot(ax=ax,label='ghi_extra')
    # ghi.reset_index(drop=True).iloc[3000:3100].plot(ax=ax, label='ghi')
    # ax.legend()
    
    
    
    LWdn_method = 'clearness'
    input_dict = {}
    
    if LWdn_method == 'dTsky':
        # Calculate LWdn using sky temperature
        input_dict['Tdb'] = d['N6_Tdb']
        
    else:
        # Calculate LWdn using cloudiness, which is estimated from clearness index
        input_dict['Tdb'] = d['N6_Tdb']
        input_dict['RH'] = d['N8_RH']
        
        input_dict['ghi_extra'] = ghi_extra
        input_dict['ghi'] = ghi
    
    LWdn = calc_LWdn(input_dict, 
                     method=LWdn_method)
    
    d['N12_HorizontalInfraredRadiationIntensity'] = LWdn
    
    d['N13_GlobalHorizontalRadiation'] = ghi.values
    
    
    
    
    
    
    # The radiation values represent here the end-of-hour values    
    
    d['N14_DirectNormalRadiation'] = df.loc[:, 'DNI'].values
    
    d['N15_DiffuseHorizontalRadiation'] = df.loc[:,'DHI'].values
    
    
    # 4 - Illuminance
    d['N16_GlobalHorizontalIlluminance'] = np.zeros(n_hours) # 999999 * np.ones(n_hours)
    
    d['N17_DirectNormalIlluminance'] = np.zeros(n_hours) # 999999 * np.ones(n_hours)
    
    d['N18_DiffuseHorizontalIlluminance'] = np.zeros(n_hours) # 999999 * np.ones(n_hours)
    
    d['N19_ZenithIlluminance'] = np.zeros(n_hours) # 9999 * np.ones(n_hours)
    
    
    # 6 - meteorological, wind and clouds
    
    d['N20_WindDirection'] = df.loc[:,'WDIR'].values
    
    d['N21_WindSpeed'] = df.loc[:,'WS'].values
    
    
    
    # second code version
    
    N_np = calc_N_from_LWdn(LWdn, d['N6_Tdb'], d['N7_Tdp'])
        
    # Energy Plus should use OpaqueSkyCover if LWdn is not available
    # However, IDA-ICE seems to read cloudines from TotalSkyCover column
    
    # IDA-ICE uses cloudiness in percent, i.e. 0-100 %
    # The epw files have tenths, i.e. 0-10
    
    d['N22_TotalSkyCover'] = N_np
    d['N23_OpaqueSkyCover'] = N_np
    
    d['N24_Visibility'] = 9999 * np.ones(n_hours)
    
    d['N25_CeilingHeight'] = 99999 * np.ones(n_hours)
    
    
    
    # 2 - not included in data source and uncertainty
    
    d['N26_PresentWeatherObservation'] = 9 * np.ones(n_hours)
    
    d['N27_PresentWeatherCodes'] = 999999999 * np.ones(n_hours)
    
    
    # 4 - rain
    
    d['N28_PrecipitableWater'] = 999 * np.ones(n_hours)
    
    d['N29_AerosolOpticalDepth'] = 0.999 * np.ones(n_hours)
    
    d['N30_SnowDepth'] = 999 * np.ones(n_hours)
    
    d['N31_DaysSinceLastSnowfall'] = 99 * np.ones(n_hours)
    
    
    # 2 - not included in the data source and uncertainty
    
    d['N32_Albedo'] = 0.2 * np.ones(n_hours) # 999 * np.ones(n_hours)
    
    d['N33_LiquidPrecipitationDepth'] = np.zeros(n_hours) # 0.068 * np.ones(n_hours)
    
    # d['N34_NotUsed_LiquidPrecipitationQuantity'] = np.nan(n_hours)
    
    
    
    X = pd.DataFrame(data=d)
    
    
    ## Write everything to a text file
    
    fname = os.path.join(output_folder,
                         file.replace('csv','epw'))
    
    # Header
    with open(fname, 'w') as f:
        
        for line in header_rows:
            
            s = line + '\n'
            f.write(s)
        
    # Data
        
    X.to_csv(fname,
             float_format='%.3f',
             header=False,
             index=False,
             mode='a',
             lineterminator='\n')
    
    














