New Delhi Next-Hour Traffic Probe Count Forecasting



Final Year Project for the BSc (Hons) in Computer Science.



Project overview



This project develops an end-to-end machine-learning workflow for forecasting the next hour's aggregated traffic probe count for ten selected roads in New Delhi. The raw source consists of daily GeoJSON traffic-probe files covering 11-30 August 2024. Segment-level observations are aggregated to road-date-hour records, temporal forecasting features are created, and three approaches are compared under chronological evaluation: a current-hour persistence baseline, Random Forest Regressor, and Gradient Boosting Regressor.



Random Forest was selected as the final model. On the final chronological holdout it achieved MAE 6,915.02, RMSE 12,052.55, and R² 0.9912, reducing MAE by 52.10% and RMSE by 53.42% relative to the persistence baseline.



The saved Random Forest preprocessing-and-model pipeline is integrated into a Streamlit historical-evaluation prototype.



Important interpretation



The project predicts aggregated road-level probe activity. It does not claim to predict an exact unique-vehicle count or a direct congestion percentage. Segment-level probe counts are summed for each selected named road, so the resulting road-level probe\_count should be interpreted as aggregated segment-level probe activity.



Dataset



Source: New Delhi Traffic Probe \& Analytics 2024 by Ryan Madhuwala and Parv Mittal on Kaggle:



https://www.kaggle.com/datasets/rawsi18/new-delhi-traffic-probe-and-analytics-2024



The original raw dataset is not stored in this repository. Download it from Kaggle and place it in the local raw-data location expected by the extraction scripts, or update PROJECT\_FOLDER in the scripts for your environment.



Main files



00\_Abstract\_Evidence\_Check.py - verifies raw archive coverage and structure.



01\_Create\_Selected\_Roads\_Dataset.py - extracts the ten selected roads and creates the hourly processed dataset.



02\_Next\_Hour\_Traffic\_Forecasting\_Colab.ipynb - feature engineering, chronological split, model comparison, final Random Forest model and holdout predictions.



03\_Exploratory\_Data\_Analysis.ipynb - data-quality checks and exploratory traffic profiles.



04\_Model\_Validation.ipynb - residual, hourly-error and road-level validation.



05\_Model\_Interpretation\_and\_Robustness.ipynb - grouped feature importance and expanding-window chronological validation.



06\_Prototype\_Readiness\_Test.ipynb - application readiness and ten-road functional checks.



app.py - Streamlit historical-evaluation application.



requirements.txt - Python package versions used by the final prototype.



Key output files



selected\_delhi\_roads\_hourly.csv



random\_forest\_next\_hour\_traffic\_model.joblib



model\_comparison\_results.csv



random\_forest\_test\_predictions.csv



random\_forest\_per\_road\_metrics.csv



feature\_importance\_results.csv



feature\_importance\_detailed.csv



rolling\_validation\_results.csv



prototype\_readiness\_test\_result.csv



prototype\_ten\_road\_test\_results.csv



Run the Streamlit prototype



Place these four files in the same application folder:



app.py



requirements.txt



random\_forest\_next\_hour\_traffic\_model.joblib



selected\_delhi\_roads\_hourly.csv



Then run:



pip install -r requirements.txt

streamlit run app.py



The prototype operates on historical observations from the processed dataset. It is not connected to a live traffic feed.



Environment



The final prototype requirements are pinned in requirements.txt, including scikit-learn 1.6.1. Use the pinned versions when loading the saved Joblib model.



Project status



Implementation and documentation completed. Final dissertation submission remains subject to supervisor review and approval.

