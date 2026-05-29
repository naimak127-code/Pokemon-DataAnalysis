# Pokémon Analytics Dashboard

A Streamlit-based data analysis and machine learning dashboard built using the Pokémon dataset.

## Project Overview

This project performs exploratory data analysis on Pokémon characteristics such as type, HP, attack, defense, speed, generation, base total, and legendary status. It also includes a machine learning model that predicts whether a Pokémon is likely to be legendary.

## Features

- Dataset overview
- Missing value handling
- Pokémon type analysis
- Attack distribution visualization
- HP vs Attack analysis
- Legendary vs Non-Legendary comparison
- Base Total by Generation analysis
- Correlation heatmap
- Top 10 strongest Pokémon
- Random Forest legendary prediction model
- Feature importance analysis
- Pokémon image display using sprite links

## Technologies Used

- Python
- Streamlit
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Machine Learning Model

The project uses a Random Forest Classifier to predict whether a Pokémon is legendary based on:

- HP
- Attack
- Defense
- Special Attack
- Special Defense
- Speed
- Base Total

Model accuracy achieved in the notebook: **93.17%**

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/pokemon-analytics-dashboard.git

2. Move into the project folder:
cd pokemon-analytics-dashboard

3. Install dependencies:
pip install -r requirements.txt

4. Run the Streamlit app:
python -m streamlit run app.py


Dataset
The dataset contains 801 Pokémon records and 41 attributes.

Project Purpose

This project was developed for a Data Mining and Warehousing course to demonstrate data cleaning, exploratory data analysis, visualization, and machine learning implementation.


