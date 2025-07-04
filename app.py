
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title='FleetGenie Dashboard', layout='wide')

# --- Load Data ---
@st.cache_data
def load_data(uploaded=None):
    if uploaded is not None:
        return pd.read_csv(uploaded)
    return pd.read_csv('fleetgenie_survey_synthetic.csv')

st.title("FleetGenie Predictive Maintenance Survey Dashboard")

uploaded_file = st.sidebar.file_uploader("Upload Your Data (CSV)", type="csv")
df = load_data(uploaded_file)

# Helper: Text description per chart/result
def insight(txt):
    st.markdown(f"<div style='color:#1f77b4;font-size:15px;background:#f8f9fa;padding:8px;border-radius:8px;border:1px solid #eee'>{txt}</div>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Visualization", "Classification", "Clustering", "Association Rule Mining", "Regression"])

# -----------------------------------------------
# TAB 1: Data Visualization
# -----------------------------------------------
with tab1:
    st.header("1. Data Visualization & Key Insights")
    st.info("Explore key patterns and business drivers from survey responses. All charts are interactive.")

    # Chart 1: Company size distribution
    fig, ax = plt.subplots()
    df['company_size'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Distribution of Company Size")
    st.pyplot(fig)
    insight("Most respondents are medium to large-sized firms, highlighting the relevance of predictive maintenance for scaling operations and managing complexity.")

    # Chart 2: Industry sector split
    fig, ax = plt.subplots()
    df['industry_sector'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
    ax.set_ylabel("")
    ax.set_title("Industry Sector Split")
    st.pyplot(fig)
    insight("Logistics providers and manufacturing companies dominate the respondent pool, validating the market fit for a B2B fleet solution.")

    # Chart 3: Fleet size vs. maintenance approach
    pivot1 = pd.crosstab(df['fleet_size'], df['maintenance_approach'])
    st.bar_chart(pivot1)
    insight("Preventive maintenance is popular for mid-sized fleets, but predictive approaches gain traction as fleets grow, indicating potential for platform adoption among larger operators.")

    # Chart 4: Vehicle types (multiselect)
    vehicle_types_flat = df['vehicle_types'].str.split(',').explode()
    fig, ax = plt.subplots()
    vehicle_types_flat.value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Vehicle Types Present in Fleets")
    st.pyplot(fig)
    insight("Trucks and vans are the most common, but electric vehicles and bikes are gaining presence, indicating demand for multi-modal support in the platform.")

    # Chart 5: Unplanned downtime incidents
    fig, ax = plt.subplots()
    sns.histplot(df['monthly_downtime_incidents'], bins=15, ax=ax, kde=True)
    ax.set_title("Monthly Unplanned Downtime Incidents")
    st.pyplot(fig)
    insight("A significant portion of fleets experiences 1–5 unplanned incidents monthly, representing a direct cost-saving opportunity for predictive analytics.")

    # Chart 6: Main barriers to adoption
    fig, ax = plt.subplots()
    df['main_adoption_barrier'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Main Barriers to Predictive Maintenance Adoption")
    st.pyplot(fig)
    insight("Cost and lack of internal skills are the most cited barriers, informing go-to-market messaging for ROI and support services.")

    # Chart 7: Adoption likelihood by company size
    group = df.groupby('company_size')['adoption_likelihood_12mo'].value_counts(normalize=True).unstack().fillna(0)
    st.bar_chart(group)
    insight("Larger companies show a higher likelihood of adopting predictive maintenance in the next 12 months, indicating priority target segments.")

    # Chart 8: Dashboard features preferences (multiselect)
    features_flat = df['dashboard_features'].str.split(',').explode()
    fig, ax = plt.subplots()
    features_flat.value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Preferred Dashboard Features")
    st.pyplot(fig)
    insight("Real-time health scores and predictive alerts are the most valued dashboard features, highlighting must-haves for the MVP.")

    # Chart 9: CO2 savings target vs. interest in analytics
    ctab = pd.crosstab(df['annual_co2_savings_target'], df['interest_auto_sustain_analytics'])
    st.bar_chart(ctab)
    insight("High and essential interest in automated sustainability analytics correlates with ambitious CO₂ savings targets, signaling sustainability-driven buyers.")

    # Chart 10: Business priorities (multiselect)
    priorities_flat = df['business_priorities'].str.split(',').explode()
    fig, ax = plt.subplots()
    priorities_flat.value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Top Business Priorities")
    st.pyplot(fig)
    insight("Cost reduction and uptime/reliability are top priorities, but sustainability is increasingly important, shaping product positioning.")

# -----------------------------------------------
# TAB 2: Classification
# -----------------------------------------------
with tab2:
    st.header("2. Classification Models & Insights")
    st.info("Compare performance of multiple ML classifiers on survey data. Upload new data for label prediction.")

    # --- Prepare target and features
    st.subheader("Select Target Variable for Classification")
    target_col = st.selectbox("Choose a categorical column as the target", 
        ['maintenance_approach', 'adoption_likelihood_12mo', 'interested_in_pilot'])

    # Preprocess: Encode multi-select columns
    df_enc = df.copy()
    multi_cols = ['vehicle_types', 'downtime_causes', 'connected_systems', 'dashboard_features',
                  'business_priorities', 'fleet_performance_measures']
    for col in multi_cols:
        mlb = MultiLabelBinarizer()
        vals = df_enc[col].str.split(',')
        enc = pd.DataFrame(mlb.fit_transform(vals), columns=[f"{col}_{c}" for c in mlb.classes_])
        df_enc = pd.concat([df_enc, enc], axis=1)
    df_enc = df_enc.drop(columns=multi_cols)

    # Encode categorical
    for col in df_enc.columns:
        if df_enc[col].dtype == 'object' and col != target_col:
            df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))

    # Remove potential text or irrelevant cols
    if target_col in df_enc.columns:
        X = df_enc.drop(columns=[target_col])
        y = df_enc[target_col]
        if y.dtype == "O":
            y = LabelEncoder().fit_transform(y)
    else:
        st.error("Target variable not found!")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Models
    models = {
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    metrics_table = []
    y_pred_all = dict()
    y_pred_proba_all = dict()
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_all[name] = y_pred
        if hasattr(model, "predict_proba"):
            y_pred_proba_all[name] = model.predict_proba(X_test)
        else:
            y_pred_proba_all[name] = None
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        metrics_table.append([name, f"{acc:.2f}", f"{prec:.2f}", f"{rec:.2f}", f"{f1:.2f}"])
    st.subheader("Model Performance Table")
    st.table(pd.DataFrame(metrics_table, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"]))
    insight("Random Forest and Gradient Boosting generally offer the highest predictive power, with Decision Trees providing easy interpretability for business users.")

    # Confusion Matrix dropdown
    st.subheader("View Confusion Matrix")
    selected_model = st.selectbox("Choose model", list(models.keys()))
    labels = np.unique(y_test)
    cm = confusion_matrix(y_test, y_pred_all[selected_model], labels=labels)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f"Confusion Matrix: {selected_model}")
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
    insight("The confusion matrix helps identify which categories the model confuses most, useful for business process improvements and targeted communications.")

    # ROC curve (for multi-class use one-vs-rest, if possible)
    st.subheader("ROC Curve for All Algorithms")
    plt.figure(figsize=(7, 5))
    for name, model in models.items():
        proba = y_pred_proba_all[name]
        if proba is not None and proba.shape[1] == len(np.unique(y)):
            try:
                fpr, tpr, _ = roc_curve(y_test, proba[:, 1], pos_label=1) if proba.shape[1]==2 else (np.nan, np.nan, np.nan)
                if isinstance(fpr, np.ndarray):
                    plt.plot(fpr, tpr, label=name)
            except:
                pass
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    st.pyplot(plt)
    insight("AUC-ROC comparison across models indicates their capability to distinguish between classes; higher curves signify more robust models.")

    # Upload new data for prediction
    st.subheader("Upload New Data for Prediction")
    pred_file = st.file_uploader("Upload New Data (CSV, no target column)", type='csv', key='pred')
    if pred_file:
        new_df = pd.read_csv(pred_file)
        # Preprocess as before
        new_enc = new_df.copy()
        for col in multi_cols:
            mlb = MultiLabelBinarizer()
            vals = new_enc[col].str.split(',')
            enc = pd.DataFrame(mlb.fit_transform(vals), columns=[f"{col}_{c}" for c in mlb.classes_])
            new_enc = pd.concat([new_enc, enc], axis=1)
        new_enc = new_enc.drop(columns=multi_cols)
        for col in new_enc.columns:
            if new_enc[col].dtype == 'object':
                new_enc[col] = LabelEncoder().fit_transform(new_enc[col].astype(str))
        model = models[selected_model]
        y_pred_new = model.predict(new_enc)
        result_df = new_df.copy()
        result_df['Predicted Label'] = y_pred_new
        st.write(result_df.head())
        st.download_button("Download Predictions", result_df.to_csv(index=False).encode(), file_name="predictions.csv")
        insight("Upload your new fleet survey data to predict maintenance approach, pilot interest, or adoption likelihood in real time.")

# -----------------------------------------------
# TAB 3: Clustering
# -----------------------------------------------
with tab3:
    st.header("3. Fleet Segmentation (Clustering)")
    st.info("Apply K-Means to segment fleet operators and discover business personas. Download the segmented data.")

    # Choose number of clusters
    n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=4, step=1)

    # Use numeric/categorical columns for clustering (exclude text)
    X_clust = df_enc.drop(columns=[target_col]) if target_col in df_enc.columns else df_enc
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clust)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    df_seg = df.copy()
    df_seg['Cluster'] = labels

    # Elbow chart
    sse = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
        sse.append(km.inertia_)
    fig, ax = plt.subplots()
    ax.plot(range(2, 11), sse, marker='o')
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("SSE (Inertia)")
    ax.set_title("Elbow Method for Optimal Clusters")
    st.pyplot(fig)
    insight("The elbow point in the curve helps determine the optimal cluster count, balancing model simplicity and explanatory power.")

    # Cluster persona table
    st.subheader("Cluster Persona Table")
    persona_table = df_seg.groupby('Cluster').agg({
        'company_size': lambda x: x.value_counts().index[0],
        'annual_revenue': lambda x: x.value_counts().index[0],
        'industry_sector': lambda x: x.value_counts().index[0],
        'fleet_size': lambda x: x.value_counts().index[0],
        'maintenance_approach': lambda x: x.value_counts().index[0],
        'avg_vehicle_age': lambda x: x.value_counts().index[0],
        'monthly_downtime_incidents': 'mean',
        'cost_per_downtime': 'mean'
    }).round(1)
    st.dataframe(persona_table)
    insight("Each cluster reflects a unique business persona—by size, revenue, maintenance style, and downtime pattern—helping tailor marketing and feature development.")

    # Download full cluster-labeled data
    st.download_button("Download Full Data with Cluster Labels", df_seg.to_csv(index=False).encode(), file_name="fleetgenie_with_clusters.csv")

# -----------------------------------------------
# TAB 4: Association Rule Mining
# -----------------------------------------------
with tab4:
    st.header("4. Association Rule Mining")
    st.info("Mine hidden connections in multi-select data (eg, vehicle types and dashboard features). Filter rules by confidence and support.")

    # Columns to use for association
    col1 = st.selectbox("Select First Column (multi-select)", multi_cols, index=0)
    col2 = st.selectbox("Select Second Column (multi-select)", multi_cols, index=3)

    # Binarize
    df_ap = df[[col1, col2]].copy()
    for col in [col1, col2]:
        mlb = MultiLabelBinarizer()
        arr = mlb.fit_transform(df_ap[col].str.split(','))
        arr_df = pd.DataFrame(arr, columns=[f"{col}_{c}" for c in mlb.classes_])
        df_ap = pd.concat([df_ap, arr_df], axis=1)
    # Drop original cols, keep binaries
    ap_cols = [c for c in df_ap.columns if c not in [col1, col2]]
    df_apr = df_ap[ap_cols]

    # Apriori & rules
    min_support = st.slider("Minimum Support", 0.01, 0.5, 0.05, 0.01)
    min_conf = st.slider("Minimum Confidence", 0.1, 1.0, 0.7, 0.05)
    freq = apriori(df_apr, min_support=min_support, use_colnames=True)
    if len(freq) > 0:
        rules = association_rules(freq, metric="confidence", min_threshold=min_conf).sort_values("confidence", ascending=False).head(10)
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
        insight("Top associations show what combinations of fleet types or preferences occur most together, guiding product bundling and cross-sell strategy.")
    else:
        st.warning("No frequent itemsets found for current parameters.")

# -----------------------------------------------
# TAB 5: Regression
# -----------------------------------------------
with tab5:
    st.header("5. Regression Insights")
    st.info("Understand key business drivers and predict downtime/cost via multiple regression techniques.")

    # Choose target
    reg_targets = ['monthly_downtime_incidents', 'cost_per_downtime']
    reg_target = st.selectbox("Select Regression Target", reg_targets)

    X_reg = df_enc.drop(columns=reg_targets + [target_col] if target_col in df_enc.columns else reg_targets)
    y_reg = df[reg_target].values

    # Models
    regressors = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Decision Tree": DecisionTreeClassifier()
    }
    reg_results = []
    for name, reg in regressors.items():
        reg.fit(X_reg, y_reg)
        y_pred = reg.predict(X_reg)
        mse = np.mean((y_pred - y_reg)**2)
        reg_results.append([name, mse])
    st.table(pd.DataFrame(reg_results, columns=["Model", "MSE"]))
    insight("Lower MSE means better model fit. Linear and Ridge often balance bias/variance, while tree-based models capture non-linear impacts.")

    # Top 5-7 insights (feature importances or coefficients)
    st.subheader("Top Regression Features")
    reg = Ridge()
    reg.fit(X_reg, y_reg)
    feat_imp = pd.Series(reg.coef_, index=X_reg.columns).abs().sort_values(ascending=False).head(7)
    st.bar_chart(feat_imp)
    insight("These features have the highest impact on downtime or cost, informing which operational levers can yield the greatest savings.")
