# Neuroscan

  DEMENTIA PREDICTION PROJECT
  Dataset: OASIS Longitudinal MRI Dataset (dementia_data.csv)
  Target:  Group (Nondemented / Demented / Converted)
  Task:    Multi-class Classification


PROJECT OVERVIEW
----------------
This project predicts dementia status using longitudinal MRI brain scan data.
Three classes exist:
  - Nondemented : No cognitive impairment
  - Demented    : Diagnosed with dementia
  - Converted   : Was nondemented but converted to demented in a later visit

MODELS USED
-----------
Classical ML : Random Forest, SVM, Decision Tree, Gradient Boosting,
               Logistic Regression, KNN, XGBoost (if installed)
Deep Learning: LSTM, GRU (sequence models using visit-level longitudinal data)

STEPS FOR RUN THIS PROJECT
-------------------------------------------
3. Steps
     1) place dementia_data.csv in the project folder;
     2) run python dementia_prediction_project.py to reproduce all models, metrics, and figures;
     3) run python -m streamlit run app.py to launch NeuroScan;
     4) open neuroscan.html in any browser for the offline tool.

5. Run this code first then run streamlit
    1) python -m pip install streamlit scikit-learn pandas numpy matplotlib seaborn
    2) python -m streamlit run dementia_prediction_project.py
    3) python -m streamlit run app.py (open another terminal and run this command)


