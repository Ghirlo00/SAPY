#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 18:18:03 2024

@author: stefanoghirlandi
"""

import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QLabel, QComboBox, QPushButton, QVBoxLayout, QMessageBox

#%% Define functions to be exported

"""
MODULE: DATA ANALYSIS FUNCTIONS
===============================

This module contains functions for various data analysis techniques, including 
linear regression, Mahalanobis distance for outlier detection, and bar plotting 
for lifestyle emissions.

"""

def lin_reg(X, y):
    """
    LINEAR REGRESSION
    -----------------

    Implements a basic linear regression to find the relationship between an 
    independent variable `X` and a dependent variable `y`.

    Parameters
    ----------
    X : numpy.ndarray
        The independent variable.
    y : numpy.ndarray
        The dependent variable.

    Returns
    -------
    fitted_values : numpy.ndarray
        The predicted values from the linear regression.
    residuals : numpy.ndarray
        The residuals, which are the differences between the actual and fitted values.
    slope : float
        The slope of the regression line.
    intercept : float
        The intercept of the regression line.

    Notes
    -----
    This function calculates the slope and intercept of the best-fit line, then 
    computes the predicted values and residuals. It does not use any specialized 
    linear regression libraries.

    """
    # Calculate the mean of X and y
    X_mean = np.mean(X)
    y_mean = np.mean(y)

    # Calculate the slope (beta) and intercept (alpha) of the regression line
    slope = np.sum((X - X_mean) * (y - y_mean)) / np.sum((X - X_mean)**2)
    intercept = y_mean - slope * X_mean

    # Calculate the fitted values (predicted values)
    fitted_values = slope * X + intercept

    # Calculate the residuals (difference between actual and fitted values)
    residuals = y - fitted_values

    return fitted_values, residuals, slope, intercept


def mahalanobis(data, threshold=0.95):
    """
   MAHALANOBIS DISTANCE AND OUTLIERS
   ---------------------------------

   Calculate the Mahalanobis distance and identify outliers based on a specified 
   threshold.

   Parameters
   ----------
   data : DataFrame
       A DataFrame containing the variables for Mahalanobis distance calculation.
   threshold : float, optional
       The percentile threshold for determining outliers (default is 0.95).

   Returns
   -------
   DataFrame
       The original data with added columns for Mahalanobis distance and outlier identification.

   Notes
   -----
   This function computes the Mahalanobis distance for each observation and uses 
   a chi-square threshold to classify outliers. It has been optimized to improve 
   performance on large datasets.
   """

    covariance_matrix = np.cov(data, rowvar=False)
    cov_inv = np.linalg.inv(covariance_matrix)

    mean_values = np.mean(data, axis=0)

    diff = data - mean_values
    mahalanobis_distances = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

    chi2_threshold = chi2.ppf(threshold, df=data.shape[1])

    data = data.copy()  # Create a copy to avoid SettingWithCopyWarning
    data['Mahalanobis Distance'] = mahalanobis_distances
    data['Outlier'] = mahalanobis_distances > chi2_threshold

    return data


def lifestyle_bar_plot(emissions, target_house, selected_country):
    """
    LIFESTYLE EMISSIONS BAR PLOT
    ----------------------------

    Creates a bar plot comparing emissions for different regions with a specified 
    household emissions target, and highlights the selected country in the plot.

    Parameters
    ----------
    emissions : pandas.Series
        A series representing the emissions for different regions, where the index 
        contains region names and the values represent the emissions in tCO2e cap-1.
       
    target_house : float
        The emissions target in tCO2e cap-1 for households (e.g., the 1.5°C target).
       
    selected_country : str
        The name of the country (region) that should be highlighted in the plot.

    Returns
    -------
    None
        Displays the bar plot. If the selected country is 'IT', the plot is saved 
        as 'Task_7-a.png'.

    Notes
    -----
    The bar plot visually represents regions' emissions, showing segments for 
    emissions below and above the target. The selected country's label is bolded 
    and colored red.
   """
    
    regions = emissions.index
    below_target = np.minimum(emissions, target_house)
    above_target = np.maximum(emissions - target_house, 0)

    fig, ax = plt.subplots(figsize=(14, 8))
    bar_width = 0.6

    ax.bar(regions, below_target, width=bar_width, label='Below 1.5°C Target', color='#69b3a2', edgecolor='black')
    ax.bar(regions, above_target, width=bar_width, bottom=below_target, label='Above 1.5°C Target', color='#ff9999', edgecolor='black')

    ax.axhline(y=target_house, color='red', linestyle='--', linewidth=2, label='Target Household Emissions')
    ax.set_xlabel('Regions', fontdict={'fontsize': 15, 'fontweight': 'bold'})
    ax.set_ylabel('Emissions (tCO2e cap-1)', fontdict={'fontsize': 15, 'fontweight': 'bold'})

    plt.xticks(rotation=45, ha='right', fontsize=13)

    # Highlight the selected country in the plot
    for label in ax.get_xticklabels():
        if label.get_text() == selected_country:
            label.set_fontsize(14)
            label.set_fontweight('bold')
            label.set_color('red')

    ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.14, 0.99), frameon=True)

    # Integrate the project logo
    logo = Image.open("lifestyles_logo.png")
    resized_logo = logo.resize((70, 80), Image.LANCZOS).convert("RGBA")
    logo_array = np.array(resized_logo)
    fig.figimage(logo_array, xo=50, yo=20, zorder=10)

    plt.tight_layout()
    plt.savefig('Task_7-a.png', dpi=150, format='png', bbox_inches='tight')
    plt.show()
    plt.close()

class CountrySelectionWindow(QDialog):
    """
    COUNTRY SELECTION WINDOW
    ------------------------

    Provides a dialog window with a dropdown menu for selecting a country to highlight 
    in data visualizations, such as a lifestyle emissions bar plot. This dialog stays on 
    top of other windows to facilitate a streamlined user experience.

    Parameters
    ----------
    countries : list of str
        List of available country names to display in the dropdown menu.
    selected_country : str
        The initially selected country shown in the dropdown menu.

    Methods
    -------
    init_ui()
        Configures the user interface elements, including the dropdown menu and OK button.
    
    on_ok()
        Updates the `selected_country` attribute with the user's choice, displays a 
        confirmation message, and closes the dialog.

    Notes
    -----
    This class inherits from `QDialog`, providing modal behavior, and uses PyQt5 
    for the user interface. The OK button is styled with a blue background and white 
    text for improved visibility.
    """
        
    def __init__(self, countries, selected_country):
        super().__init__()
        self.countries = countries
        self.selected_country = selected_country
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Select Country to Highlight")
        self.setWindowFlag(Qt.WindowStaysOnTopHint)  # Keep the window on top

        # Create a label to instruct the user
        label = QLabel("Please select a country to highlight:", self)

        # Create a dropdown menu with the list of countries
        self.dropdown = QComboBox(self)
        self.dropdown.addItems(self.countries)
        self.dropdown.setCurrentText(self.selected_country)

        # Create an OK button
        button_ok = QPushButton('OK', self)
        button_ok.setStyleSheet("background-color: blue; color: white; font-weight: bold;")
        button_ok.clicked.connect(self.on_ok)

        # Create a layout and add widgets
        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.dropdown)
        layout.addWidget(button_ok)

        self.setLayout(layout)

    def on_ok(self):
        self.selected_country = self.dropdown.currentText()
        QMessageBox.information(self, "Selection", f"You selected: {self.selected_country}")
        self.accept() 
