# Iris Dataset Exploratory Data Analysis (EDA) using OOP in Python

This project provides a complete exploratory data analysis (EDA) of the famous **Iris dataset** using an **Object-Oriented Programming (OOP)** approach in Python. The analysis includes summary statistics, correlation matrix, visualizations (such as boxplots and PCA), and classification using a Random Forest classifier. The visualizations are saved as images in the project directory.

## Features

- **Object-Oriented Design**: The entire analysis is encapsulated in a Python class, following OOP principles for better code reusability and structure.
- **Data Preparation**: The dataset is split into training and testing sets, and features are scaled.
- **Dimensionality Reduction**: Principal Component Analysis (PCA) is performed and visualized to reduce the dimensionality of the dataset.
- **Classification**: A Random Forest classifier is used to predict the species, and its performance is evaluated using a classification report and confusion matrix.
- **Image Saving**: The project automatically saves key visualizations (Boxplots, PCA plots) in a `plots` directory.

## Project Structure

- **`main.py`**: The main Python script containing the `IrisEDA` class that handles all the data processing, visualization, and classification tasks.
- **`plots/`**: A folder where the generated plots (boxplot and PCA plot) are saved as `.png` images.

## Requirements

To run this project, you need the following Python libraries installed:

```bash
pip install pandas seaborn matplotlib scikit-learn