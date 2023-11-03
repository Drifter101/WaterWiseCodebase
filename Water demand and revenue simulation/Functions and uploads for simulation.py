import numpy as np
import matplotlib.pyplot as plt
import copy
from functools import reduce
import pandas as pd
import scipy.stats as stats
import math
from fitter import Fitter
from collections import Counter
import seaborn as sns
import time
from scipy.stats import logistic, gamma

########################################### IMPORT DATA #########################################################################

file_path = r'E:\_Series annuales\Basedata_clean.csv'
df = pd.read_csv(file_path, parse_dates=['FECHA'])
df['Month'] = df['FECHA'].dt.to_period('M')

########################################### DICCTIONARY ######################################################################

# Create a dictionary where the keys are the 'TIPUS_VIVENDA' values and the values are the corresponding data subsets
data_dict = {tipus: df[df['TIPUS_VIVENDA'] == tipus] for tipus in df['TIPUS_VIVENDA'].unique()}

# Access the data subset where TIPUS_VIVENDA = B
df_b = data_dict['B']
df_c = data_dict['C']
df_d = data_dict['D']
df_e = data_dict['E']


# Parametros: 
trench_limits = [6, 9, 15, 18, float('inf')]
trench_rates = [0.6180, 1.2361, 1.8541, 2.4719, 3.0899]
new_trench_limits = [9, float('inf')]
new_trench_rates = [0.8924,1.7848]

# Funciones
def apply_trench_processing(df, trench_limits, trench_rates):
    def determine_trench_and_process(consumed_water):
        for i in range(len(trench_limits)):
            if consumed_water <= trench_limits[i]:
                trench_number = i + 1
                if trench_number == 1:
                    result = consumed_water * trench_rates[0]
                elif trench_number == 2:
                    result = (
                        trench_limits[0] * trench_rates[0]
                        + trench_rates[1] * (consumed_water - trench_limits[0])
                    )
                elif trench_number == 3:
                    result = (
                        trench_limits[0] * trench_rates[0]
                        + trench_rates[1] * (trench_limits[1] - trench_limits[0])
                        + trench_rates[2] * (consumed_water - trench_limits[1])
                    )
                elif trench_number == 4:
                    result = (
                        trench_limits[0] * trench_rates[0]
                        + trench_rates[1] * (trench_limits[1] - trench_limits[0])
                        + trench_rates[2] * (trench_limits[2] - trench_limits[1])
                        + trench_rates[3] * (consumed_water - trench_limits[2])
                    )
                else:
                    result = (
                        trench_limits[0] * trench_rates[0]
                        + trench_rates[1] * (trench_limits[1] - trench_limits[0])
                        + trench_rates[2] * (trench_limits[2] - trench_limits[1])
                        + trench_rates[3] * (trench_limits[3] - trench_limits[2])
                        + trench_rates[4] * (consumed_water - trench_limits[3])
                    )

                return trench_number, result

        trench_number = len(trench_limits)
        result = 0  # Provide a default value for the result if the consumed water exceeds all trench limits

        return trench_number, result

    df['trench_number'], df['processed_water'] = zip(
        *df.apply(lambda row: determine_trench_and_process(row['CONSUMO_REAL']), axis=1)
    )
    return df

def apply_commercial_trench_processing(df, trench_limits, trench_rates):
    def determine_commercial_trench_and_process(consumed_water):
        for i in range(len(trench_limits)):
            if consumed_water <= trench_limits[i]:
                trench_number = i + 1
                if trench_number == 1:
                    result = consumed_water * trench_rates[0]
                else:  # When trench_number == 2
                    result = (
                        trench_limits[0] * trench_rates[0]
                        + trench_rates[1] * (consumed_water - trench_limits[0])
                    )
                return trench_number, result

        trench_number = len(trench_limits)
        result = 0  # Provide a default value for the result if the consumed water exceeds all trench limits

        return trench_number, result

    df['trench_number'], df['processed_water'] = zip(
        *df.apply(lambda row: determine_commercial_trench_and_process(row['CONSUMO_REAL']), axis=1)
    )
    return df

def simulate_top_water_consumption(df, iterations, fscale):
    # Initialize an empty DataFrame to store the simulated values
    simulated_data = pd.DataFrame(columns=['Month', 'CONSUMO_REAL'])

    # Loop over the months
    for month in df['Month'].unique():
        # Filter the data for the current month
        data = df[df['Month'] == month]['CONSUMO_REAL']

        # Fit the gamma distribution to the data with modified scale parameter
        shape, loc, scale = gamma.fit(data, floc=0, fscale=fscale)

        # Generate random values from the gamma distribution
        simulated_values = gamma.rvs(shape, loc=loc, scale=scale, size=iterations)

        # Create a DataFrame with the simulated values for the current month
        simulated_month_data = pd.DataFrame({'Month': np.repeat(month, iterations),
                                             'CONSUMO_REAL': simulated_values})

        # Append a random sample of the simulated values to the overall DataFrame
        simulated_data = simulated_data.append(simulated_month_data.sample(n=iterations, replace=True), ignore_index=True)

    # Sort the simulated data by 'CONSUMO_REAL'
    simulated_data = simulated_data.sort_values('CONSUMO_REAL').reset_index(drop=True)

    # Calculate the quantiles for assigning 'Habitantes' values
    quantiles = simulated_data['CONSUMO_REAL'].quantile([0.92])

    # Assign 'Habitantes' values based on quantiles
    simulated_data['Habitantes'] = 5  # Highest 8%
    simulated_data.loc[simulated_data['CONSUMO_REAL'] <= quantiles[0.92], 'Habitantes'] = 0  # The rest

    # Filter only the top 8% of the values (those with 'Habitantes' equal to 5)
    simulated_data = simulated_data[simulated_data['Habitantes'] == 5]

    # Return the simulated data
    return simulated_data
def simulate_water_consumption(df, iterations, fscale):
    # Initialize an empty DataFrame to store the simulated values
    simulated_data = pd.DataFrame(columns=['Month', 'CONSUMO_REAL'])

    # Loop over the months
    for month in df['Month'].unique():
        # Filter the data for the current month
        data = df[df['Month'] == month]['CONSUMO_REAL']

        # Fit the gamma distribution to the data with modified scale parameter
        shape, loc, scale = gamma.fit(data, floc=0, fscale=fscale)  # Adjust the fscale parameter as desired

        # Generate random values from the gamma distribution
        simulated_values = gamma.rvs(shape, loc=loc, scale=scale, size=iterations)

        # Create a DataFrame with the simulated values for the current month
        simulated_month_data = pd.DataFrame({'Month': np.repeat(month, iterations),
                                             'CONSUMO_REAL': simulated_values})

        # Append a random sample of the simulated values to the overall DataFrame
        simulated_data = simulated_data.append(simulated_month_data.sample(n=iterations, replace=True), ignore_index=True)

    # Sort the simulated data by 'Simulated_CONSUMO_REAL'
    simulated_data = simulated_data.sort_values('CONSUMO_REAL').reset_index(drop=True)

    # Calculate the quantiles for assigning 'Habitantes' values
    quantiles = simulated_data['CONSUMO_REAL'].quantile([0.22, 0.5, 0.73, 0.92])

    # Assign 'Habitantes' values based on quantiles
    simulated_data['Habitantes'] = 1  # Lowest 22%
    simulated_data.loc[simulated_data['CONSUMO_REAL'] > quantiles[0.22], 'Habitantes'] = 2  # Next 28%
    simulated_data.loc[simulated_data['CONSUMO_REAL'] > quantiles[0.5], 'Habitantes'] = 3  # Next 23%
    simulated_data.loc[simulated_data['CONSUMO_REAL'] > quantiles[0.73], 'Habitantes'] = 4  # Next 19%
    simulated_data.loc[simulated_data['CONSUMO_REAL'] > quantiles[0.92], 'Habitantes'] = 5  # Highest 8%

    # Return the simulated data
    return simulated_data


def process_data(group):
    first_split, second_split = split_data(group)
    first_split = apply_trench_processing(first_split, trench_limits, trench_rates)
    second_split = apply_new_trench_processing(second_split, new_trench_limits, new_trench_rates)
    return pd.concat([first_split, second_split]).reset_index()

def monte_carlo_process(df, df_name):
    df = df.copy()
    df_grouped = df.groupby(df['Month'])
    df_splits = [process_data(group) for _, group in df_grouped]
    df_processed = pd.concat(df_splits)
    df_sum = df_processed.groupby('Month')['processed_water'].sum().reset_index()
    return df_sum












