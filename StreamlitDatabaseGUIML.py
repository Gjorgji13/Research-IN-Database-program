# StreamlitFullInDBML_LQO_merged_fixed_FINAL.py
import time
import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text, inspect
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from io import BytesIO
from pandas import ExcelWriter

st.set_page_config(page_title="In-DB ML Explorer — Scalable Analytics", layout="wide")
st.title("📊 In-Database ML Explorer — Regression & Classification + Scalability")

# -------------------------
# Sidebar: DB connection
# -------------------------
st.sidebar.header("Database connection")
db_host = st.sidebar.text_input("Host", value="localhost")
db_port = st.sidebar.text_input("Port", value="5000")
db_user = st.sidebar.text_input("User", value="root")
db_pass = st.sidebar.text_input("Password", value="1497", type="password")

# list databases
try:
    engine_tmp = create_engine(f"mysql+mysqlconnector://{db_user}:{db_pass}@{db_host}:{db_port}/")
    with engine_tmp.connect() as conn_tmp:
        dbs = pd.read_sql("SHOW DATABASES", conn_tmp)["Database"].tolist()
        dbs = [d for d in dbs if d not in ("information_schema", "mysql", "performance_schema", "sys")]
except Exception as e:
    st.sidebar.error(f"Could not list databases: {e}")
    st.stop()

db_choice = st.sidebar.selectbox("Select database", dbs)
DB_URL = f"mysql+mysqlconnector://{db_user}:{db_pass}@{db_host}:{db_port}/{db_choice}"

try:
    engine = create_engine(DB_URL)
    inspector = inspect(engine)
    st.sidebar.success(f"Connected to `{db_choice}` ✅")
except Exception as e:
    st.sidebar.error(f"Connection failed: {e}")
    st.stop()

# -------------------------
# Fetch tables and load schema preview
# -------------------------
with engine.connect() as conn:
    try:
        tables_df = pd.read_sql(
            f"SELECT TABLE_NAME FROM information_schema.tables WHERE TABLE_SCHEMA = '{db_choice}'", conn
        )
        table_list = tables_df.iloc[:, 0].tolist()
    except Exception as e:
        st.error(f"Could not list tables: {e}")
        st.stop()

selected_table = st.sidebar.selectbox("Select table", table_list)
st.subheader(f"Preview: `{selected_table}` (first 10 rows)")
with engine.connect() as conn:
    try:
        # Load small preview for schema detection and UI setup
        df_schema_preview = pd.read_sql(f"SELECT * FROM `{selected_table}` LIMIT 100", conn)
        st.dataframe(df_schema_preview.head(10))
    except Exception as e:
        st.error(f"Could not preview table: {e}")
        st.stop()

# -------------------------
# Scalability Options
# -------------------------
st.sidebar.header("Scalability Options")
use_sampling = st.sidebar.checkbox("Use Random Sample for Training?", value=False)
sample_size_perc = st.sidebar.slider("Sample Size (%)", 0.5, 10.0, 5.0) / 100.0

pk_col = None
try:
    pk_info = inspector.get_pk_constraint(selected_table)
    pk_cols = pk_info.get("constrained_columns", [])
    if pk_cols:
        pk_col = pk_cols[0]
except Exception:
    pk_col = None

# -------------------------
# Auto-detect query log table / ML setup
# -------------------------
auto_query_features = False

if "query_text" in df_schema_preview.columns and "execution_time" in df_schema_preview.columns:
    auto_query_features = True
    feature_cols = ["query_length", "num_spaces", "num_tables"]
    target_col = "execution_time"
    problem_type = "regression"
    st.write("Detected query log table. Auto-generated features will be computed on sampled data.")
else:
    numeric_cols = df_schema_preview.select_dtypes(include=np.number).columns.tolist()
    object_cols = df_schema_preview.select_dtypes(include=["object", "datetime", "category"]).columns.tolist()
    all_cols = df_schema_preview.columns.tolist()

    # Sidebar controls
    st.sidebar.header("ML setup")
    if numeric_cols:
        default_target = numeric_cols[-1]
    else:
        default_target = all_cols[-1] if all_cols else None

    if default_target not in all_cols:
        default_target = all_cols[0] if all_cols else None

    target_col = st.sidebar.selectbox("Target column (Y)", all_cols,
                                      index=all_cols.index(default_target) if default_target else 0)
    # Use df_schema_preview to determine problem type
    problem_type = "regression" if pd.api.types.is_numeric_dtype(df_schema_preview[target_col]) else "classification"
    st.sidebar.markdown(f"**Detected problem type:** `{problem_type}`")
    default_features = [c for c in all_cols if c != target_col]
    feature_cols = st.sidebar.multiselect("Feature columns (X)", all_cols, default=default_features)

# -------------------------
# Sidebar ML model choice
# -------------------------
if not auto_query_features:
    if problem_type == "regression":
        model_choice = st.sidebar.selectbox("Model", ["Linear Regression", "Random Forest", "XGBoost"])
    else:
        model_choice = st.sidebar.selectbox("Model", ["Logistic Regression", "Random Forest", "XGBoost"])
else:
    model_choice = st.sidebar.selectbox("Model", ["Linear Regression", "Random Forest", "XGBoost"])

test_size = st.sidebar.slider("Test size (%)", 10, 50, 20) / 100.0
random_state = int(st.sidebar.number_input("Random seed", value=42))
apply_scaling = st.sidebar.checkbox("Apply feature scaling", value=True)
train_btn = st.sidebar.button("Train & Predict")


# -------------------------
# Utilities (Remains the same)
# -------------------------
def build_preprocessor(df, features, apply_scaling=True):
    num_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler() if apply_scaling else "passthrough", num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
    return ColumnTransformer(transformers, remainder="drop") if transformers else None

def get_feature_names(preprocessor):
    """Retrieves feature names after preprocessing, including OneHotEncoder names."""
    feature_names = []
    # Loop through the transformers in the ColumnTransformer
    for name, trans, cols in preprocessor.transformers_:
        if trans == 'passthrough':
            feature_names.extend(cols)
        elif hasattr(trans, 'get_feature_names_out'):
            # This handles both StandardScaler (returns original names) and OneHotEncoder
            # (returns expanded names, e.g., 'City_London')
            feature_names.extend(trans.get_feature_names_out(cols))
        elif hasattr(trans, 'get_feature_names'): # Older scikit-learn compatibility
            feature_names.extend(trans.get_feature_names(cols))
        else:
            feature_names.extend(cols) # Fallback
    return feature_names

# -------------------------
# Train & predict (The core integrated logic)
# -------------------------
if train_btn:
    if not feature_cols:
        st.error("Select at least one feature column.")
        st.stop()

    # --- 1. Load Data with Sampling for Training ---
    st.info("Loading data from database...")

    if use_sampling:
        with engine.connect() as conn:
            # Get total row count for accurate sampling size
            total_rows = pd.read_sql(f"SELECT COUNT(*) FROM `{selected_table}`", conn).iloc[0, 0]
            N_sample = int(total_rows * sample_size_perc)

            # Use RAND() for sampling (note: slow on very large tables, but standard SQL)
            sample_query = f"SELECT * FROM `{selected_table}` ORDER BY RAND() LIMIT {N_sample}"
            df_ml = pd.read_sql(sample_query, conn)
            st.success(f"✅ Loaded {len(df_ml)} rows (Sample Size: {sample_size_perc * 100:.1f}%) for training.")
    else:
        # Fallback to full load (for smaller datasets)
        with engine.connect() as conn:
            df_ml = pd.read_sql(f"SELECT * FROM `{selected_table}`", conn)
            st.success(f"✅ Loaded {len(df_ml)} rows (Full Dataset) for training.")

    # --- 2. Data Preprocessing and Splitting (Uses df_ml) ---
    if auto_query_features:
        # Calculate features on the sampled data
        df_ml["query_length"] = df_ml["query_text"].apply(lambda x: len(str(x)))
        df_ml["num_spaces"] = df_ml["query_text"].apply(lambda x: str(x).count(" "))
        df_ml["num_tables"] = df_ml["query_text"].apply(lambda x: str(x).upper().count("FROM"))

    df_train_only = df_ml[feature_cols + [target_col]].dropna()
    if df_train_only.empty:
        st.error("After dropping NA, no data left for training.")
        st.stop()

    X = df_train_only[feature_cols].copy()
    y = df_train_only[target_col].copy()

    label_enc = None
    if problem_type.lower() == "classification":
        label_enc = LabelEncoder()
        y = label_enc.fit_transform(y.astype(str))

    preprocessor = build_preprocessor(df_train_only, feature_cols, apply_scaling=apply_scaling)
    X_proc = preprocessor.fit_transform(X) if preprocessor else X.values

    X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=test_size, random_state=random_state)

    # --- 3. Build Model ---
    if problem_type.lower() == "regression":
        if model_choice == "Linear Regression":
            model = LinearRegression()
        # ... (rest of model instantiation remains the same)
        elif model_choice == "Random Forest":
            model = RandomForestRegressor(n_estimators=200, random_state=random_state)
        else:
            model = XGBRegressor(n_estimators=200, max_depth=8, random_state=random_state,
                                 objective="reg:squarederror", eval_metric="rmse")
    else:
        if model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        # ... (rest of model instantiation remains the same)
        elif model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=200, random_state=random_state)
        else:
            model = XGBClassifier(n_estimators=200, max_depth=8, random_state=random_state,
                                  use_label_encoder=False, eval_metric="logloss")

    start_t = time.time()
    # 🧠 Model Training & Prediction
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)

    # Convert predictions for classification metrics
    if problem_type.lower() == "classification":
        # Simplified prediction to classes for metrics
        y_pred_test_classes = np.round(y_pred_test).astype(int)
    else:
        y_pred_test_classes = y_pred_test  # Regression

    # --- 4. Evaluation (Uses test set) ---
    st.subheader("📈 Model Evaluation Results")
    if problem_type.lower() == "regression":
        mae = mean_absolute_error(y_test, y_pred_test)
        rmse = mean_squared_error(y_test, y_pred_test) ** 0.5
        r2 = r2_score(y_test, y_pred_test)
        st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")
        st.write(f"**R² Score:** {r2:.4f}")
    else:
        acc = accuracy_score(y_test, y_pred_test_classes)
        prec = precision_score(y_test, y_pred_test_classes, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred_test_classes, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred_test_classes, average='weighted', zero_division=0)
        st.write(f"**Accuracy:** {acc:.4f}")
        st.write(f"**Precision:** {prec:.4f}")
        st.write(f"**Recall:** {rec:.4f}")
        st.write(f"**F1 Score:** {f1:.4f}")



    # -------------------------
    # Feature Importance (Required Graph 1)
    # -------------------------

    # --- Get the correct, expanded feature names ---
    if preprocessor:
        # Use the preprocessor to get the final feature list (needed due to OneHotEncoder)
        final_feature_names = get_feature_names(preprocessor)
    else:
        # Fallback if no preprocessing was necessary
        final_feature_names = list(X.columns)

    if hasattr(model, "feature_importances_") or (
            hasattr(model, 'coef_') and problem_type in ['regression', 'classification']):
        st.subheader("📊 Feature Importance/Coefficient Chart")

        importances = None

        if hasattr(model, "feature_importances_"):
            # For tree-based models (Random Forest, XGBoost)
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models (Linear Regression, Logistic Regression)
            # Note: coefficients for multiclass classification might be 2D, so flatten/take mean/select one.
            if problem_type == 'classification' and model.coef_.ndim > 1:
                # For simplicity, use the absolute average of coefficients across classes
                importances = np.mean(np.abs(model.coef_), axis=0)
            else:
                # For regression or binary classification
                importances = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)

        # CRITICAL CHECK: Ensure the length of names matches the length of importances
        if importances is not None and len(final_feature_names) == len(importances) and importances.any():
            feat_imp = pd.DataFrame({
                "Feature": final_feature_names,
                "Importance": importances
            }).sort_values("Importance", ascending=False)

            st.bar_chart(feat_imp.set_index("Feature"))
        else:
            st.info(
                "Model evaluation completed, but feature importance/coefficients were not available or did not match feature count.")
    else:
        st.info("Feature analysis is not supported for the selected model type.")

    # -------------------------
    # 🎯 Actual vs Predicted Plot (Required Graph 2 - Remains the same)
    # -------------------------
    try:
        st.subheader("🎯 Actual vs Predicted Visualization")
        viz_df = pd.DataFrame({
            "Actual": y_test,
            "Predicted": y_pred_test_classes
        })
        st.scatter_chart(viz_df)
    except Exception as e:
        st.warning(f"⚠️ Visualization failed: {e}")
    # ... (rest of the code follows) ...

    # -------------------------
    # 💥 SEPARATOR BEFORE CHUNKING (Ensures graphs load above status messages)
    # -------------------------
    st.markdown("---")
    st.subheader("💾 Full Prediction and DB Write-Back (Chunked)")

    # --- 6. Chunking for Full Prediction & Write-Back (The Scalability Solution) ---
    output_table_name = f"{selected_table}_predictions"
    chunksize = 50000

    # Preprocessor must be saved and used to transform the chunks

    if st.checkbox("Write full predictions back to database?", value=True):

        # Use a fresh connection for the iterative read/write
        with engine.connect() as conn:

            # Start chunking the FULL dataset from the DB
            for i, chunk in enumerate(pd.read_sql(f"SELECT * FROM `{selected_table}`", conn, chunksize=chunksize)):
                st.info(f"Processing chunk {i + 1} of size {len(chunk)}...")

                # Apply auto-features if needed
                if auto_query_features:
                    chunk["query_length"] = chunk["query_text"].apply(lambda x: len(str(x)))
                    chunk["num_spaces"] = chunk["query_text"].apply(lambda x: str(x).count(" "))
                    chunk["num_tables"] = chunk["query_text"].apply(lambda x: str(x).upper().count("FROM"))

                # Preprocess and Predict on the chunk
                X_chunk = chunk[feature_cols].copy()

                # IMPORTANT: Use the preprocessor fitted on the sampled data
                X_proc_chunk = preprocessor.transform(X_chunk)

                # Generate predictions
                predictions = model.predict(X_proc_chunk)

                # Convert classification predictions back if applicable
                if problem_type.lower() == "classification" and label_enc:
                    predictions = label_enc.inverse_transform(predictions.astype(int))

                # Store results in the chunk
                chunk["Predicted_" + target_col] = predictions

                # Write chunk back to DB
                if i == 0:
                    # Create the table on the first chunk
                    chunk.to_sql(output_table_name, conn, if_exists="replace", index=False)
                else:
                    # Append subsequent chunks
                    chunk.to_sql(output_table_name, conn, if_exists="append", index=False)

        st.success(f"✅ Full predictions successfully written to new table: `{output_table_name}`")

    st.success("Analysis complete.")
