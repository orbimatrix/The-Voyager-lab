# Kepler Exoplanet Data Analysis 

This repository contains data, notebooks, models and a small Gradio demo for exoplanet vs false-positive classification built from Kepler TCEs and FPP labels.

## What’s included (high-level)
- Notebooks with analysis and model training:
  - [prediction_models.ipynb](prediction_models.ipynb) — end-to-end training, feature engineering, SMOTE balancing, model selection and evaluation.
  - [Logistic_regression.ipynb](Logistic_regression.ipynb) — additional model experiments and LR baseline.
  - [expolanet.ipynb](expolanet.ipynb), [fpp.ipynb](fpp.ipynb), [stellar_hosts.ipynb](stellar_hosts.ipynb) — data exploration and derived features.
  - [kepler_visuals.ipynb](kepler_visuals.ipynb), [stellar_visuals.ipynb](stellar_visuals.ipynb) — visualizations.
- Prediction demo and API:
  - [prediction/app.py](prediction/app.py) — Gradio demo that loads the trained objects and exposes [`predict_exoplanet`](prediction/app.py).
  - [prediction/gradio.ipynb](prediction/gradio.ipynb) — interactive notebook used to run/share the Gradio app.
- Saved model and preprocessors (used by the demo):
  - [prediction/exoplanet_model.pkl](prediction/exoplanet_model.pkl) — trained model (best ensemble from training notebook).
  - [prediction/scaler.pkl](prediction/scaler.pkl) — fitted scaler used for preprocessing.
  - [prediction/feature_names.pkl](prediction/feature_names.pkl) — list of feature columns and engineered features.

## Data
- Primary TCE table: [q1_q17_dr25_tce_2025.02.03_04.32.18.csv](q1_q17_dr25_tce_2025.02.03_04.32.18.csv)
- FPP labels: [FPP_table.csv](FPP_table.csv)
- Kepler objects & stellar catalogs: [Kepler_objects.csv](Kepler_objects.csv), [STELLARHOSTS.csv](STELLARHOSTS.csv)
- Example usages & small tables are saved in the notebooks.

## Prediction / Demo
- The demo uses the function [`predict_exoplanet`](prediction/app.py) and the saved artifacts: [`model`](prediction/app.py) → [prediction/exoplanet_model.pkl](prediction/exoplanet_model.pkl), [`scaler`](prediction/app.py) → [prediction/scaler.pkl](prediction/scaler.pkl), and [`feature_names`](prediction/app.py) → [prediction/feature_names.pkl](prediction/feature_names.pkl).
- Run locally:
  1. Ensure Python environment with dependencies (pandas, numpy, scikit-learn, joblib, gradio, xgboost if needed).
  2. From the `prediction/` folder run:
     ```sh
     python app.py
     ```
  3. The Gradio UI will open (or use the notebook prediction/gradio.ipynb).

## Reproduce training
- Open [prediction_models.ipynb](http://_vscodecontentref_/21) — it contains:
  - Data loading and merge of TCE + FPP labels
  - Feature selection: e.g., `['tce_period','tce_time0bk','tce_impact','tce_duration','tce_depth','tce_model_snr','tce_prad','tce_eqt','tce_insol','tce_steff','tce_slogg','tce_sradius']`
  - Engineered features: `temp_radius_ratio`, `period_duration_ratio`, `snr_depth_product`
  - Train/test split, SMOTE balancing, individual model training (RandomForest, XGBoost, etc.), stacking/selection and ROC-AUC evaluation.
  - Final objects saved via joblib: see code that writes `exoplanet_model.pkl`, [scaler.pkl](http://_vscodecontentref_/22), [feature_names.pkl](http://_vscodecontentref_/23).

## Key model & features notes
- Binary target: 1 = Exoplanet (fpp_prob < 0.5), 0 = False Positive.
- Evaluation metrics printed in notebooks include accuracy, classification report, confusion matrix and ROC-AUC.
- Example prediction pipeline (used in the app):
  1. Create DataFrame from inputs.
  2. Add engineered features (same order as [feature_names](http://_vscodecontentref_/24)).
  3. Scale using [scaler](http://_vscodecontentref_/25).
  4. Predict with the saved [model](http://_vscodecontentref_/26) and (if available) [predict_proba](http://_vscodecontentref_/27) for confidence.

## Quick examples (from demo)
- Example inputs used in the demo and notebooks are included under the Gradio examples in [app.py](http://_vscodecontentref_/28) and [gradio.ipynb](http://_vscodecontentref_/29).

## Development & contribution
- To retrain or change feature set, edit and run [prediction_models.ipynb](http://_vscodecontentref_/30).
- To change the UI or inputs, edit [app.py](http://_vscodecontentref_/31) and re-run the demo.

## License & data attribution
- Uses publicly available NASA Kepler data from the NASA Exoplanet Archive. See notebooks for data source comments and attribution.

## Contact
- For questions about model internals, open [app.py](http://_vscodecontentref_/32) and [prediction_models.ipynb](http://_vscodecontentref_/33) — the main logic and parameters are documented inline.