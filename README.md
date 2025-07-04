
# FleetGenie Predictive Maintenance Survey Dashboard

This Streamlit dashboard allows interactive exploration, analysis, and predictive modeling on logistics fleet survey data for the FleetGenie business idea.

## Features
- **Data Visualization**: 10+ charts with business insights from survey data.
- **Classification**: Compare KNN, Decision Tree, Random Forest, and Gradient Boosting. Confusion matrix, ROC curve, prediction on new data.
- **Clustering**: K-means with elbow plot and business persona table. Download labeled data.
- **Association Rule Mining**: Apriori mining of multi-select columns with parameter sliders.
- **Regression**: Linear, Ridge, Lasso, and Decision Tree regression on downtime/cost drivers.

## How to Use

1. **Clone this repo or download all files** (including `app.py`, `fleetgenie_survey_synthetic.csv`, `requirements.txt`, and `README.md`).
2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
3. **Run locally:**  
   ```bash
   streamlit run app.py
   ```
4. **Or deploy to [Streamlit Cloud](https://streamlit.io/cloud):**  
   - Push these files to a GitHub repo (no folders).
   - Create a new app, link your repo, and set `app.py` as the entry point.
   - You can pull data from your GitHub using raw URLs (edit the file loader if needed).

## Notes
- To analyze your own data, use the sidebar upload tool or place your CSV in the same directory.
- **All outputs are interactive, downloadable, and labeled for business insights.**

---
Dashboard built by [YourName/FleetGenie] – July 2025.
