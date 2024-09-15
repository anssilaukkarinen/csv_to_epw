# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 22:16:51 2024

@author: Anssi Laukkarinen

The purpose of this code is to read in hourly climate data from csv
files and export it to epw climate files.


Input climate files are from the Finnish Meteorological Institute.
https://www.ilmatieteenlaitos.fi/rakennusten-energialaskennan-testivuosi

Input file column names:
STEP;YEAR;MON;DAY;HOUR;TEMP;RH;WS;WDIR;GHI;DHI;DNI



Information on the epw file format:
https://bigladdersoftware.com/epx/docs/23-2/auxiliary-programs/energyplus-weather-file-epw-data-dictionary.html#energyplus-weather-file-epw-data-dictionary



"""

import os
import numpy as np
import pandas as pd

import pvlib

import helper


root_folder = r'C:\Users\laukkara\github\csv_to_epw'

input_folder = os.path.join(root_folder,
                            'input')

output_folder = os.path.join(root_folder,
                             'output')

if not os.path.exists(output_folder):
    os.makedirs(output_folder)





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
    
    timestamps_np = helper.get_epw_timestamps(df)
    
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
                    f'COMMENTS 1,"{file}"',
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
    
    d['N7_Tdp'] = helper.calc_Tdp(df.loc[:,'TEMP'].values, 
                                  df.loc[:,'RH'].values)
    
    d['N8_RH'] = df.loc[:,'RH'].values
    
    d['N9_Patm'] = 101325.0*np.ones(n_hours)
    
    
    ## 2 - extraterrestrial radiation for calculating clearness index
    # Calculate extraterrestrial beam radiation
    dni_extra = pvlib.irradiance.get_extra_radiation( \
                    datetime_or_doy=df.index.shift(freq="-30min"))
    
    
    
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
    
    LWdn = helper.calc_LWdn(input_dict, 
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
    
    N_np = helper.calc_N_from_LWdn(LWdn, d['N6_Tdb'], d['N7_Tdp'])
        
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
                         'epw_' + file.replace('csv','epw'))
    
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
    
    














