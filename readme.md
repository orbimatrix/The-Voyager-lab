# Kepler Exoplanet Data Analysis

This project analyzes data from NASA's Kepler Space Telescope mission to explore and visualize characteristics of exoplanets and their host stars.

## Dataset

The analysis uses the following key datasets:
- [Kepler_objects.csv](Kepler_objects.csv) - Main Kepler Objects of Interest (KOI) dataset
- [q1_q17_dr25_tce_2025.02.03_04.32.18.csv](q1_q17_dr25_tce_2025.02.03_04.32.18.csv) - TCE data
- [STELLARHOSTS.csv](STELLARHOSTS.csv) - Host star properties

## Analysis & Visualizations

The project includes several key visualizations:

1. **Disposition Distribution**
   - Breakdown of KOIs by disposition (Confirmed, Candidate, False Positive)
   - Visualized using a count plot

2. **Planet Properties**
   - Distribution of planetary radii (in Earth radii)
   - Planet radius vs orbital period relationships
   - Equilibrium temperature distribution

3. **Stellar Properties** 
   - Host star temperature distribution
   - Stellar parameter analysis

4. **Feature Correlations**
   - Correlation matrix of key planetary and stellar parameters:
     - Orbital period
     - Planet radius
     - Equilibrium temperature
     - Insolation flux
     - Transit depth
     - Transit duration
     - Signal-to-noise ratio
     - Stellar temperature
     - Surface gravity
     - Stellar radius

## Requirements

```python
pandas
numpy
matplotlib
seaborn
```

## Usage

The analysis is contained in Jupyter notebooks:
- [kepler_visuals.ipynb](kepler_visuals.ipynb) - Main visualization notebook
- [stellar_visuals.ipynb](stellar_visuals.ipynb) - Stellar host analysis
- [prediction_models.ipynb](prediction_models.ipynb) - ML models for exoplanet classification

## Data Processing

Key data cleaning steps:
- Removal of NULL values
- Handling missing values in key parameters
- Filtering for required columns
- Log scaling where appropriate for visualization

## License

This project uses publicly available NASA Kepler data from the NASA Exoplanet Archive.