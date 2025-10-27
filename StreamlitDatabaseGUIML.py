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


# Optional explainability dependency - imported lazily
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

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

# optional WHERE filter executed in SQL
st.sidebar.header("Optional SQL Filter")
where_clause = st.sidebar.text_input("SQL WHERE clause (e.g. region='North' AND year>=2020)", value="")

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

# Try to detect primary key
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
    st.write("Detected query log table. Auto-generated features will be computed using SQL when possible.")
else:
    # Fallback to generic ML
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
# Utilities
# -------------------------
def build_preprocessor(df, features, apply_scaling=True):
    num_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler() if apply_scaling else "passthrough", num_cols))
    if cat_cols:
        # scikit-learn may expect sparse_output or sparse depending on version; we try to be compatible
        try:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
        except TypeError:
            # older sklearn versions use 'sparse' parameter
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols))
    return ColumnTransformer(transformers, remainder="drop") if transformers else None

def get_feature_names(preprocessor):
    """Retrieves feature names after preprocessing, including OneHotEncoder names."""
    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if trans == 'passthrough':
            feature_names.extend(cols)
        else:
            # Some transformers are estimators (e.g. StandardScaler) that don't change names
            if hasattr(trans, "get_feature_names_out"):
                # get_feature_names_out expects feature names in newer sklearn
                try:
                    out_names = trans.get_feature_names_out(cols)
                except Exception:
                    # if StandardScaler, it may return cols unchanged
                    out_names = cols
                feature_names.extend(list(out_names))
            elif hasattr(trans, "get_feature_names"):
                try:
                    out_names = trans.get_feature_names(cols)
                except Exception:
                    out_names = cols
                feature_names.extend(list(out_names))
            else:
                feature_names.extend(cols)
    return list(feature_names)

# -------------------------
# Train & predict (Main logic)
# -------------------------
if train_btn:
    if not feature_cols:
        st.error("Select at least one feature column.")
        st.stop()

    # --- 1. Load Data with Sampling for Training ---
    st.info("Loading data from database (with optional SQL feature generation)...")

    with engine.connect() as conn:
        if use_sampling:
            # Get total row count for sampling
            total_rows = pd.read_sql(f"SELECT COUNT(*) as cnt FROM `{selected_table}`", conn).iloc[0, 0]
            N_sample = max(1, int(total_rows * sample_size_perc))

            # If auto_query_features: compute features in SQL
            if auto_query_features:
                # CHAR_LENGTH, REPLACE, UPPER used for MySQL; adjust if DB differs
                sample_query = f"""
                SELECT *, 
                       CHAR_LENGTH(query_text) AS query_length,
                       (LENGTH(query_text) - LENGTH(REPLACE(query_text, ' ', ''))) AS num_spaces,
                       ( (LENGTH(UPPER(query_text)) - LENGTH(REPLACE(UPPER(query_text), 'FROM', ''))) / 4 ) AS num_tables
                FROM `{selected_table}` 
                {f"WHERE {where_clause}" if where_clause else ""}
                ORDER BY RAND() LIMIT {N_sample}
                """
            else:
                sample_query = f"SELECT * FROM `{selected_table}` {f'WHERE {where_clause}' if where_clause else ''} ORDER BY RAND() LIMIT {N_sample}"
            df_ml = pd.read_sql(sample_query, conn)
            st.success(f"✅ Loaded {len(df_ml)} rows (Sample Size: {sample_size_perc * 100:.1f}%) for training.")
        else:
            # Full load, but with SQL feature generation when auto_query_features
            if auto_query_features:
                full_query = f"""
                SELECT *, 
                       CHAR_LENGTH(query_text) AS query_length,
                       (LENGTH(query_text) - LENGTH(REPLACE(query_text, ' ', ''))) AS num_spaces,
                       ( (LENGTH(UPPER(query_text)) - LENGTH(REPLACE(UPPER(query_text), 'FROM', ''))) / 4 ) AS num_tables
                FROM `{selected_table}` {f"WHERE {where_clause}" if where_clause else ""}
                """
            else:
                full_query = f"SELECT * FROM `{selected_table}` {f'WHERE {where_clause}' if where_clause else ''}"
            df_ml = pd.read_sql(full_query, conn)
            st.success(f"✅ Loaded {len(df_ml)} rows (Full Dataset) for training.")

    # --- 2. Data Preprocessing and Splitting (Uses df_ml) ---
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

    # Save preprocessed data back to DB for reproducibility (small table - can be toggled)
    save_preproc = st.sidebar.checkbox("Save preprocessed training table to DB?", value=True)
    preproc_table = f"{selected_table}_preprocessed"
    if save_preproc:
        try:
            df_train_only_to_save = X.copy()
            df_train_only_to_save[target_col] = y if label_enc is None else label_enc.inverse_transform(y)
            with engine.begin() as conn:
                df_train_only_to_save.to_sql(preproc_table, conn, if_exists="replace", index=False)
            st.info(f"Preprocessed training set saved to `{preproc_table}`.")
        except Exception as e:
            st.warning(f"Could not save preprocessed table: {e}")

    # Get feature names after preprocessing (OneHot expands columns)
    if preprocessor:
        try:
            final_feature_names = get_feature_names(preprocessor)
        except Exception:
            final_feature_names = list(X.columns)
    else:
        final_feature_names = list(X.columns)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=test_size, random_state=random_state)

    # --- 3. Build Model ---
    if problem_type.lower() == "regression":
        if model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "Random Forest":
            model = RandomForestRegressor(n_estimators=200, random_state=random_state)
        else:
            model = XGBRegressor(n_estimators=200, max_depth=8, random_state=random_state,
                                 objective="reg:squarederror", eval_metric="rmse")
    else:
        if model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=200, random_state=random_state)
        else:
            model = XGBClassifier(n_estimators=200, max_depth=8, random_state=random_state,
                                  use_label_encoder=False, eval_metric="logloss")

    start_t = time.time()
    # Model Training & Prediction
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)

    # Convert predictions for classification metrics (some models return probabilities)
    if problem_type.lower() == "classification":
        try:
            if hasattr(y_pred_test, "ndim") and y_pred_test.ndim > 1 and y_pred_test.shape[1] > 1:
                y_pred_test_classes = np.argmax(y_pred_test, axis=1)
            else:
                y_pred_test_classes = np.round(y_pred_test).astype(int)
        except Exception:
            y_pred_test_classes = np.round(y_pred_test).astype(int)
    else:
        y_pred_test_classes = y_pred_test  # Regression

    # --- 4. Evaluation (Uses test set) ---
    st.subheader("📈 Model Evaluation Results")
    r2 = None
    acc = None
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
    # Feature Importance
    # -------------------------
    if hasattr(model, "feature_importances_") or (hasattr(model, 'coef_') and problem_type in ['regression', 'classification']):
        st.subheader("📊 Feature Importance/Coefficient Chart")

        importances = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            try:
                if problem_type == 'classification' and model.coef_.ndim > 1:
                    importances = np.mean(np.abs(model.coef_), axis=0)
                else:
                    importances = np.abs(model.coef_[0]) if getattr(model.coef_, "ndim", 1) > 1 else np.abs(model.coef_)
            except Exception:
                importances = None

        # Build DataFrame using final_feature_names (preprocessing-expanded)
        if importances is not None and len(final_feature_names) == len(importances):
            feat_imp = pd.DataFrame({
                "Feature": final_feature_names,
                "Importance": importances
            }).sort_values("Importance", ascending=False)

            st.dataframe(feat_imp.head(50))
            try:
                st.bar_chart(feat_imp.set_index("Feature").head(20))
            except Exception:
                pass
        else:
            try:
                fi_len = len(importances) if importances is not None else 0
            except Exception:
                fi_len = 0
            st.info(f"Feature importance unavailable or length mismatch (features: {len(final_feature_names)}, importances: {fi_len}).")
    else:
        st.info("Feature analysis is not supported for the selected model type.")

    # --- create stable feature_importance_df for downstream OLAP usage ---
    try:
        if hasattr(model, "feature_importances_") and len(final_feature_names) == len(model.feature_importances_):
            feature_importance_df = pd.DataFrame({
                "Feature": final_feature_names,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False)
        else:
            if hasattr(model, 'coef_'):
                coef_arr = np.mean(np.abs(model.coef_), axis=0) if getattr(model.coef_, "ndim", 1) > 1 else np.abs(model.coef_)
                coef_arr = np.array(coef_arr).flatten()
                if len(final_feature_names) == len(coef_arr):
                    feature_importance_df = pd.DataFrame({"Feature": final_feature_names, "Importance": coef_arr}).sort_values("Importance", ascending=False)
                else:
                    feature_importance_df = pd.DataFrame({"Feature": final_feature_names, "Importance": [0]*len(final_feature_names)})
            else:
                feature_importance_df = pd.DataFrame({"Feature": final_feature_names, "Importance": [0]*len(final_feature_names)})
    except Exception:
        feature_importance_df = pd.DataFrame({"Feature": final_feature_names, "Importance": [0]*len(final_feature_names)})

    # -------------------------
    # Optional: SHAP explainability (lightweight if available)
    # -------------------------
    if SHAP_AVAILABLE:
        try:
            st.subheader("🔍 SHAP Explainability (optional)")
            # shap expects original feature space (not scaled/encoded mismatch may happen)
            # This is a best-effort; skip if incompatible
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_test)
            st.pyplot(shap.plots.bar(shap_values, show=False))
        except Exception as e:
            st.info(f"SHAP explanation skipped (error): {e}")

    # -------------------------
    # 🎯 Actual vs Predicted Visualization
    # -------------------------
    try:
        st.subheader("🎯 Actual vs Predicted Visualization")
        viz_df = pd.DataFrame({
            "Actual": np.array(y_test).flatten(),
            "Predicted": np.array(y_pred_test_classes).flatten()
        })
        st.dataframe(viz_df.head(100))
        st.scatter_chart(viz_df)
    except Exception as e:
        st.warning(f"⚠️ Visualization failed: {e}")

    # -------------------------
    # === NEW: OLAP ANALYSIS MODULE ===
    # -------------------------
    st.markdown("---")
    st.header("🔎 OLAP Analysis Module")
    st.write("Perform pivot/group-by (OLAP-style) analysis on the same table. Choose either in-memory pivot (fast for small datasets) or SQL GROUP BY (scalable).")

    # provide the option to reload (full) data into df_olap (careful with huge tables)
    olap_load_method = st.radio("OLAP data source", ["From sample/load (current df_ml)", "Direct SQL GROUP BY (scalable)"], index=0)

    if olap_load_method == "From sample/load (current df_ml)":
        df_olap = df_ml.copy()
        st.info("Using the dataset already loaded for training (may be sampled).")
    else:
        st.info("SQL GROUP BY mode: queries executed directly in the database.")
        df_olap = None  # will use SQL for group-by operations

    # Pivot (single or multi-dim) with safeguards
    st.subheader("Pivot / Aggregate")
    if olap_load_method == "From sample/load (current df_ml)":
        # Only show columns that exist and are 1-dimensional
        cols_available = df_olap.columns.tolist()
        default_dim = [cols_available[0]] if cols_available else []
        dims = st.multiselect("Group by dimension(s)", cols_available, default=default_dim)
        numeric_cols = df_olap.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            measure = st.selectbox("Select measure (numeric)", numeric_cols)
            agg_func = st.selectbox("Aggregation function", ["sum", "mean", "count", "max", "min"])
            if dims and measure:
                # ensure dims is a flat list of valid column names
                dims = [d for d in dims if d in df_olap.columns]
                try:
                    pivot = pd.pivot_table(df_olap, values=measure, index=dims, aggfunc=agg_func)
                    st.dataframe(pivot.reset_index().head(200))
                    try:
                        st.bar_chart(pivot.reset_index().set_index(dims[0]).head(50))
                    except Exception:
                        pass
                except Exception as e:
                    st.error(f"Pivot creation failed: {e}")
        else:
            st.info("No numeric columns available for aggregation.")
    else:
        # SQL GROUP BY mode
        group_col = st.selectbox("Group by column (SQL)", df_schema_preview.columns.tolist())
        numeric_cols_sql = df_schema_preview.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols_sql:
            measure_sql = st.selectbox("Measure (SQL)", numeric_cols_sql)
            agg_sql = st.selectbox("SQL Aggregation", ["SUM", "AVG", "COUNT", "MAX", "MIN"])
            sql_query = f"SELECT {group_col}, {agg_sql}({measure_sql}) AS aggregated_value FROM `{selected_table}`"
            if where_clause:
                sql_query += f" WHERE {where_clause}"
            sql_query += f" GROUP BY {group_col} ORDER BY aggregated_value DESC LIMIT 1000"
            try:
                df_grouped = pd.read_sql(sql_query, engine)
                st.dataframe(df_grouped)
                try:
                    st.bar_chart(df_grouped.set_index(group_col))
                except Exception:
                    pass
            except Exception as e:
                st.error(f"SQL GROUP BY failed: {e}")

    # Drill-down
    st.subheader("Drill-down")
    try:
        if olap_load_method == "From sample/load (current df_ml)":
            dd_col = st.selectbox("Column to drill by", df_olap.columns.tolist())
            dd_values = df_olap[dd_col].dropna().unique().tolist()
            selected_value = st.selectbox("Value to drill into", dd_values)
            if st.button("Show drill-down rows"):
                df_drill = df_olap[df_olap[dd_col] == selected_value]
                st.dataframe(df_drill.head(500))
        else:
            # SQL-based drill-down
            dd_col_sql = st.selectbox("SQL Drill-by column", df_schema_preview.columns.tolist())
            # get list of top values
            sample_vals_q = f"SELECT {dd_col_sql} AS v, COUNT(*) AS cnt FROM `{selected_table}`"
            if where_clause:
                sample_vals_q += f" WHERE {where_clause}"
            sample_vals_q += f" GROUP BY {dd_col_sql} ORDER BY cnt DESC LIMIT 200"
            vals_df = pd.read_sql(sample_vals_q, engine)
            if not vals_df.empty:
                chosen = st.selectbox("Pick value to drill", vals_df["v"].astype(str).tolist())
                if st.button("Run SQL drill-down"):
                    drill_q = f"SELECT * FROM `{selected_table}` WHERE {dd_col_sql} = %s"
                    if where_clause:
                        drill_q = f"SELECT * FROM `{selected_table}` WHERE {where_clause} AND {dd_col_sql} = %s"
                    try:
                        df_dr = pd.read_sql(drill_q, engine, params=(chosen,))
                        st.dataframe(df_dr.head(1000))
                    except Exception as e:
                        st.error(f"Drill-down SQL failed: {e}")
            else:
                st.info("No distinct values found for drill-down.")
    except Exception as e:
        st.info(f"Drill-down UI skipped (error): {e}")

    # OLAP results can be used as features for a new model — optional quick example
    st.subheader("Use aggregated OLAP features for ML (quick demo)")
    if st.button("Generate simple aggregated features (group by one column)"):
        try:
            if olap_load_method == "From sample/load (current df_ml)":
                gb_col = st.selectbox("Group-by column for aggregated features", df_olap.columns.tolist(), key="gb1")
                agg_col = st.selectbox("Numeric column to aggregate", df_olap.select_dtypes(include=[np.number]).columns.tolist(), key="agg1")
                agg_df = df_olap.groupby(gb_col)[agg_col].agg(['sum', 'mean', 'count']).reset_index()
                st.dataframe(agg_df.head(200))
                st.info("Aggregated features generated. You can download or join them back to the original table offline.")
            else:
                gb_col = st.selectbox("Group-by column (SQL)", df_schema_preview.columns.tolist(), key="gb2")
                agg_col = st.selectbox("Numeric column (SQL)", df_schema_preview.select_dtypes(include=[np.number]).columns.tolist(), key="agg2")
                agg_q = f"SELECT {gb_col}, SUM({agg_col}) AS sum_{agg_col}, AVG({agg_col}) AS avg_{agg_col}, COUNT(*) AS cnt FROM `{selected_table}`"
                if where_clause:
                    agg_q += f" WHERE {where_clause}"
                agg_q += f" GROUP BY {gb_col} ORDER BY cnt DESC LIMIT 10000"
                agg_df_sql = pd.read_sql(agg_q, engine)
                st.dataframe(agg_df_sql.head(200))
                st.info("SQL aggregated features ready. Consider exporting or joining them in DB for model training.")
        except Exception as e:
            st.error(f"Aggregation for ML failed: {e}")

    # -------------------------
    # Save model metadata to DB
    # -------------------------
    st.markdown("---")
    st.subheader("💾 Model Metadata & Save")
    model_save = st.checkbox("Save model metadata to DB?", value=True)
    if model_save:
        try:
            metric_val = float(r2) if (problem_type == 'regression' and r2 is not None) else (float(acc) if acc is not None else None)
            meta = {
                "model_name": model_choice,
                "problem_type": problem_type,
                "train_rows": int(len(df_train_only)),
                "test_size": test_size,
                "random_seed": int(random_state),
                "timestamp": pd.Timestamp.now(),
                "metric": metric_val
            }
            meta_df = pd.DataFrame([meta])
            with engine.begin() as conn:
                meta_df.to_sql("model_metadata", conn, if_exists="append", index=False)
            st.success("Model metadata saved to table `model_metadata`.")
        except Exception as e:
            st.warning(f"Could not save model metadata: {e}")

    # -------------------------
    # 💾 Full Prediction and DB Write-Back (Chunked)
    # -------------------------
    st.markdown("---")
    st.subheader("💾 Full Prediction and DB Write-Back (Chunked)")

    output_table_name = f"{selected_table}_predictions"
    chunksize = int(st.sidebar.number_input("Chunk size for write-back", value=50000, step=5000))

    if st.checkbox("Write full predictions back to database?", value=True):
        with engine.begin() as conn:
            i = 0
            try:
                for chunk in pd.read_sql(f"SELECT * FROM `{selected_table}`", conn, chunksize=chunksize):
                    i += 1
                    st.info(f"Processing chunk {i} of size {len(chunk)}...")
                    if auto_query_features and "query_text" in chunk.columns:
                        chunk["query_length"] = chunk["query_text"].apply(lambda x: len(str(x)))
                        chunk["num_spaces"] = chunk["query_text"].apply(lambda x: str(x).count(" "))
                        chunk["num_tables"] = chunk["query_text"].apply(lambda x: str(x).upper().count("FROM"))

                    # guard: ensure feature columns exist in chunk
                    missing = [c for c in feature_cols if c not in chunk.columns]
                    if missing:
                        st.error(f"Missing feature columns in chunk: {missing}")
                        break

                    X_chunk = chunk[feature_cols].copy()
                    if preprocessor:
                        # preprocessor expects the same columns - handle missing / unseen categories by filling NAs
                        X_chunk = X_chunk.reindex(columns=feature_cols)
                        X_proc_chunk = preprocessor.transform(X_chunk)
                    else:
                        X_proc_chunk = X_chunk.values

                    preds = model.predict(X_proc_chunk)
                    if problem_type.lower() == "classification" and label_enc is not None:
                        try:
                            preds_out = label_enc.inverse_transform(np.array(preds).astype(int))
                        except Exception:
                            preds_out = preds
                    else:
                        preds_out = preds

                    chunk["Predicted_" + target_col] = preds_out

                    if i == 1:
                        chunk.to_sql(output_table_name, conn, if_exists="replace", index=False)
                    else:
                        chunk.to_sql(output_table_name, conn, if_exists="append", index=False)
                st.success(f"✅ Full predictions successfully written to new table: `{output_table_name}`")
            except Exception as e:
                st.error(f"Chunked write-back failed: {e}")

    st.success("Analysis complete.")
