# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 09:36:36 2023

@author: Nicol
"""
import seaborn as sns
import time
import os
# Directory path
directory = r'E:\_Series annuales'
############################################## Scenario Baseline ################################################
monte_carlo_iterations = 1000  # Number of iterations for the Monte Carlo simulation
total_sums = []  # List to store the total of each simulation

start_time = time.time()

for i in range(monte_carlo_iterations):
    simulated_data_b = simulate_water_consumption(df_b, iterations=3000, fscale=3.9)  #38737
    simulated_data_c = simulate_water_consumption(df_c, iterations=2500, fscale=0.5)
    simulated_data_d = simulate_water_consumption(df_d, iterations=1550, fscale=1.5)
    simulated_data_e = simulate_water_consumption(df_e, iterations=359, fscale=1.18)

    baseline_bill_b = apply_trench_processing(simulated_data_b, trench_limits, trench_rates)
    baseline_bill_c = apply_trench_processing(simulated_data_c, trench_limits, trench_rates)
    baseline_bill_d = apply_trench_processing(simulated_data_d, trench_limits, trench_rates)
    baseline_bill_e = apply_trench_processing(simulated_data_e, trench_limits, trench_rates)

    baseline_bill_b_sum = baseline_bill_b.groupby('Month')['processed_water'].sum().reset_index()
    baseline_bill_c_sum = baseline_bill_c.groupby('Month')['processed_water'].sum().reset_index()
    baseline_bill_d_sum = baseline_bill_d.groupby('Month')['processed_water'].sum().reset_index()
    baseline_bill_e_sum = baseline_bill_e.groupby('Month')['processed_water'].sum().reset_index()

    scenario_base = pd.concat([baseline_bill_b_sum['Month'], baseline_bill_b_sum['processed_water'],
                            baseline_bill_c_sum['processed_water'], baseline_bill_d_sum['processed_water'],
                            baseline_bill_e_sum['processed_water']], axis=1)

    scenario_base.columns = ['Month', 'processed_water_bill_b', 'processed_water_bill_c',
                          'processed_water_bill_d', 'processed_water_bill_e']

    scenario_base['Total'] = scenario_base[['processed_water_bill_b', 'processed_water_bill_c',
                                      'processed_water_bill_d', 'processed_water_bill_e']].sum(axis=1)

    total_sums.append(scenario_base['Total'].sum())

    # Print progress and estimated time left every 100 iterations
    if (i+1) % 100 == 0:
        elapsed_time = time.time() - start_time
        iterations_left = monte_carlo_iterations - (i+1)
        estimated_time_left = (elapsed_time / (i+1)) * iterations_left
        print(f"Finished {i+1}/{monte_carlo_iterations} iterations. Estimated time left: {estimated_time_left/60:.2f} minutes.")

average_total = sum(total_sums) / monte_carlo_iterations
print(f"Global average of scenario_base over {monte_carlo_iterations} simulations: {average_total}")

sums_scenario_base = total_sums
# Create a histogram and KDE of the simulation results
sns.histplot(sums_scenario_base, kde=True, bins=30)

plt.title('Escenario 1: Histograma simulación con tarifa actual')
plt.xlabel('Euros')
plt.ylabel('Iteraciones')
plt.show()

# Convert the list to a DataFrame
df_sums_scenario_base = pd.DataFrame(sums_scenario_base)
# File path
file_path = os.path.join(directory, 'sums_scenario_base.csv')
# Save DataFrame to CSV
df_sums_scenario_base.to_csv(file_path, index=False)
############################################### Scenario with new prices ################################################################
monte_carlo_iterations = 1000  # Number of iterations for the Monte Carlo simulation
total_sums = []  # List to store the total of each simulation

for _ in range(monte_carlo_iterations):
    simulated_data_b = simulate_water_consumption(df_b, iterations=3000, fscale=3.9)  #38737,
    simulated_data_c = simulate_water_consumption(df_c, iterations=2500, fscale=0.5)  #31250
    simulated_data_d = simulate_water_consumption(df_d, iterations=1550, fscale=1.5)  #19375
    simulated_data_e = simulate_water_consumption(df_e, iterations=359, fscale=1.18)  #4487 

    bill_b = apply_trench_processing(simulated_data_b.copy(), trench_limits, new_trench_rates)
    bill_c = apply_trench_processing(simulated_data_c.copy(), trench_limits, new_trench_rates)
    bill_d = apply_trench_processing(simulated_data_d.copy(), trench_limits, new_trench_rates)
    bill_e = apply_trench_processing(simulated_data_e.copy(), trench_limits, new_trench_rates)

    bill_b_sum = bill_b.groupby('Month')['processed_water'].sum().reset_index()
    bill_c_sum = bill_c.groupby('Month')['processed_water'].sum().reset_index()
    bill_d_sum = bill_d.groupby('Month')['processed_water'].sum().reset_index()
    bill_e_sum = bill_e.groupby('Month')['processed_water'].sum().reset_index()

    scenario_100 = pd.concat([bill_b_sum['Month'], bill_b_sum['processed_water'],
                            bill_c_sum['processed_water'], bill_d_sum['processed_water'],
                            bill_e_sum['processed_water']], axis=1)

    scenario_100.columns = ['Month', 'processed_water_bill_b', 'processed_water_bill_c',
                          'processed_water_bill_d', 'processed_water_bill_e']

    scenario_100['Total'] = scenario_100[['processed_water_bill_b', 'processed_water_bill_c',
                                      'processed_water_bill_d', 'processed_water_bill_e']].sum(axis=1)

    # Add the total of the current simulation to the list
    total_sums.append(scenario_100['Total'].sum())

# Calculate the average over all the simulations
average_total = sum(total_sums) / monte_carlo_iterations

print(f"Global average of scenario_100 over {monte_carlo_iterations} simulations: {average_total}")

sums_scenario_100 = total_sums
# Create a histogram and KDE of the simulation results
sns.histplot(sums_scenario_100, kde=True, bins=30,color="green")

plt.title('Escenario 1: Histograma simulación con tarifa comercial')
plt.xlabel('Euros')
plt.ylabel('Iteraciones')
plt.show()

# Convert the list to a DataFrame
df_sums_scenario_100 = pd.DataFrame(sums_scenario_100)
# File path
file_path = os.path.join(directory, 'sums_scenario_100.csv')
# Save DataFrame to CSV
df_sums_scenario_100.to_csv(file_path, index=False)

