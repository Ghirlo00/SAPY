#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:46:42 2024

@author: stefanoghirlandi
"""

import pandas as pd
import scipy.stats as sc
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from assignment_A_functions import lin_reg, mahalanobis, lifestyle_bar_plot, CountrySelectionWindow

from PyQt5.QtWidgets import QApplication, QDialog

import os

#%%                             Task 1&2 - Load Data
target_house = 2.38 # median target household emissions in 2030

interactive_mode = True  # Set to False to disable the interactive GUI

selected_country = 'IT'  # Default selection for highlighting in the plot

base_dir = os.path.dirname(os.path.abspath(__file__))

# Load Data
G20 = pd.read_csv(os.path.join(base_dir, "Input data", "G20.csv"), encoding='iso-8859-1')
footprint = pd.read_csv(os.path.join(base_dir, "Input data", "footprints.csv"))
categories = pd.read_csv(os.path.join(base_dir, "Input data", "categories.csv"))

# G20 = pd.read_csv("../SAPY_A_Stefano/Input data/G20.csv", encoding='iso-8859-1')
# footprint = pd.read_csv("../SAPY_A_Stefano/Input data/footprints.csv")
# categories = pd.read_csv("../SAPY_A_Stefano/Input data/categories.csv")

#%%                         Task 4 - Pre-process the data

# G20 group with lifestyle footprints in 2030
G20_index = G20.loc[:, 'iso3']
footprint1 = footprint[(footprint['Year'] == 2030) & (footprint['Region'].isin(G20_index))]
categories1 = categories[(categories['Year'] == 2030) & (categories['Region'].isin(G20_index))]

categories1 = categories1.pivot(index='Region', columns='Category', values='Emissions (tCO2e cap-1)')
footprint1 = footprint1.drop('Year', axis=1).set_index('Region')

#%%                      Task 5 - Perform the main analysis

# a) Calculate the gaps between the lifestyle carbon footprints and the emissions compatible with the 1.5°C target. 
footprint_gaps = footprint1.copy()
footprint_gaps['Gap'] = footprint_gaps['Emissions (tCO2e cap-1)'] - target_house

# b) Calculate CTV for each consumption category
merged_data = footprint_gaps.merge(categories1, on='Region')
correlations = merged_data.corr(method='spearman')

CTV_df = correlations['Gap'].drop('Gap').drop('Emissions (tCO2e cap-1)')
CTV_df = (CTV_df ** 2) # Square the correlations
CTV_df.columns = ['Category']  # Rename columns

CTV = CTV_df.div(CTV_df.sum())

# c) Identify the category with the largest contribution to variance 
CTV_max = CTV.idxmax()

# f) Test normality of Gap and Direct emissions
Gap_norm = sm.stats.lilliefors(merged_data['Gap'], dist='norm')[1]
DE_norm = sm.stats.lilliefors(merged_data['Direct emissions'], dist='norm')[1]

# g) Test homoscedasticity by inspecting a scatter plot
X = merged_data['Gap'].values
y = merged_data['Direct emissions'].values

fitted_values, residuals, slope, intercept = lin_reg(X, y) # as imported from Functions.py

plt.figure(figsize=(10, 6))
plt.scatter(fitted_values, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Fitted Values (Direct emissions)')
plt.ylabel('Residuals')
plt.show()
plt.close()


# h) Test the absence of bivariate outliers with the Mahalanobis distance.
data_for_mahalanobis = merged_data[['Gap', 'Direct emissions']]
result_with_mahalanobis = mahalanobis(data_for_mahalanobis)

# i) Pearson coefficient for statistical significance
pearson_corr, p_value = sc.pearsonr(merged_data['Gap'], merged_data['Direct emissions'])
    
# d) Conditional decisions based on test results
print("\n--- Conditional Decisions ---")

print("Gap normality:", f"Yes because {Gap_norm:.4f}" if Gap_norm > 0.05 else f"No because {Gap_norm:.4f}")
print("Direct emissions normality:", f"Yes because {DE_norm:.4f}" if DE_norm > 0.05 else f"No because {DE_norm:.4f}")

print("Visual check of the scatter plot for homoscedasticity")

print("Bivariate outliers detected" if any(result_with_mahalanobis['Outlier']) else "No bivariate outliers detected")

# e) Conclusions from tests
print("\n--- Conclusions ---")

normality_msg = "Both 'Gap' and 'Direct emissions' are normally distributed." if Gap_norm > 0.05 and DE_norm > 0.05 else "One or both variables are not normally distributed."
print(normality_msg)

significance_msg = f"Pearson correlation: {pearson_corr:.4f}, {'result is significant' if p_value < 0.05 else 'result is not significant'} (p = {p_value:.4g})."
print(significance_msg)


#%%          Task 6 - Export the main results to a csv

# a) Remove categories without contributions to variance
CTV_filtered = CTV[CTV > 0]
CTV_percentages = CTV_filtered * 100

# b) DataFrame with the analysis choices and a blank line
analysis_choices = pd.DataFrame({
    'Category': ['Country Group', 'Target Year', 'Type of Footprints', 'Correlation Method', '\n'],
    'Contribution to Variance (%)': ['G20', '2030', 'Lifestyle carbon footprint', 'Spearman rank correlation', '\n']
})

export_data = pd.DataFrame(CTV_percentages).reset_index()
export_data.columns = ['Category', 'Contribution to Variance (%)']

# c) Combine the analysis choices and the main data
final_export = pd.concat([analysis_choices, export_data], ignore_index=True)
final_export.to_csv('contribution.csv', index=False)

#%%             Task 7&8 - GUI and plots

#  Stacked bar plot with lifestyle carbon footprints below and above the 1.5°C target
emissions = categories1.sum(axis=1)  
emissions = emissions.sort_index() # Sort the emissions data by index (regions)
    
if interactive_mode:
    app = QApplication.instance()
    
    selected_country = 'IT'  # Initial country selection
    # Open the selection dialog
    window = CountrySelectionWindow(footprint1.index.tolist(), selected_country)
    
    # Execute the dialog and wait for it to close
    if window.exec_() == QDialog.Accepted:
        selected_country = window.selected_country
    
    # Plot after the dialog has closed and QApplication loop ends
    lifestyle_bar_plot(emissions, target_house, selected_country)
    
    app = None
    

# Plot of the pie chart
fig, ax = plt.subplots(figsize=(14, 8))

plt.pie(CTV_percentages.sort_values(),
        labels=CTV_percentages.index, 
        autopct='%1.1f%%', 
        startangle=90, 
        textprops={'fontsize': 14, 'fontweight': 'bold'})

logo = Image.open("lifestyles_logo.png")
resized_logo = logo.resize((70, 80), Image.LANCZOS).convert("RGBA")
logo_array = np.array(resized_logo)

fig.figimage(logo_array, xo=50, yo=20, zorder=10)  

plt.tight_layout()
plt.savefig('Task_7-e.png', dpi=150, format='png', bbox_inches='tight')
plt.show()
plt.close()
