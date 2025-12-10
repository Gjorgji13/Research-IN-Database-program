# StreamlitFullInDBML_LQO_Clean.py

import time
import tracemalloc
import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, inspect, text
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import sqlite3
import matplotlib.pyplot as plt

# Optional SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

st.set_page_config(page_title="In-DB ML & DB Explorer (Clean)", layout="wide")
st.title("🧭 Database Research Explorer — Query & Index Performance + In-DB ML")

# -------------------------
# Sidebar: DB Connection
# -------------------------
st.sidebar.header("Database connection")
db_host = st.sidebar.text_input("Host", value="localhost")
db_port = st.sidebar.text_input("Port", value="5000")
db_user = st.sidebar.text_input("User", value="root")
db_pass = st.sidebar.text_input("Password", value="1497", type="password")

st.sidebar.header("Options")
where_clause = st.sidebar.text_input("Optional SQL WHERE clause (e.g., year>=2020)", value="")
use_sampling = st.sidebar.checkbox("Use Random Sample for ML training?", value=False)
sample_size_perc = st.sidebar.slider("Sample size for ML (%)", 0.5, 50.0, 5.0) / 100.0
train_btn = st.sidebar.button("Train & Predict (ML)")

# -------------------------
# Connect to MySQL and list DBs
# -------------------------
try:
    engine_tmp = create_engine(f"mysql+mysqlconnector://{db_user}:{db_pass}@{db_host}:{db_port}/")
    with engine_tmp.connect() as conn_tmp:
        dbs = pd.read_sql("SHOW DATABASES", conn_tmp)["Database"].tolist()
        dbs = [d for d in dbs if d not in ("information_schema", "mysql", "performance_schema", "sys")]
except Exception as e:
    st.sidebar.error(f"Could not list databases: {e}")
    st.stop()

db_choice = st.sidebar.selectbox("Select database", dbs, key="db_choice")
DB_URL = f"mysql+mysqlconnector://{db_user}:{db_pass}@{db_host}:{db_port}/{db_choice}"

try:
    engine = create_engine(DB_URL)
    inspector = inspect(engine)
    st.sidebar.success(f"Connected to `{db_choice}` ✅")
except Exception as e:
    st.sidebar.error(f"Connection failed: {e}")
    st.stop()

# -------------------------
# Fetch tables & preview
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

selected_table = st.sidebar.selectbox("Select table", table_list, key="table_selectbox")
st.subheader(f"Preview: `{selected_table}` (first 10 rows)")
with engine.connect() as conn:
    try:
        df_preview = pd.read_sql(f"SELECT * FROM `{selected_table}` LIMIT 10", conn)
        st.dataframe(df_preview)
    except Exception as e:
        st.error(f"Preview failed: {e}")

# -------------------------
# Utilities: preprocessors
# -------------------------
def build_preprocessor(df, features, apply_scaling=True):
    num_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler() if apply_scaling else "passthrough", num_cols))
    if cat_cols:
        try:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
        except TypeError:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols))
    return ColumnTransformer(transformers, remainder="drop") if transformers else None

def get_feature_names(preprocessor):
    feature_names = []
    if preprocessor is None:
        return feature_names
    for name, trans, cols in preprocessor.transformers_:
        if trans == "passthrough":
            feature_names.extend(cols)
        else:
            try:
                if hasattr(trans, "get_feature_names_out"):
                    feature_names.extend(list(trans.get_feature_names_out(cols)))
                elif hasattr(trans, "get_feature_names"):
                    feature_names.extend(list(trans.get_feature_names(cols)))
                else:
                    feature_names.extend(cols)
            except Exception:
                feature_names.extend(cols)
    return list(feature_names)

# -------------------------
# Module 1: Query Performance & Exploration
# -------------------------
st.header("1️⃣ Query Performance & Exploration")
query_input = st.text_area(
    "Enter SELECT query",
    value=f"SELECT * FROM `{selected_table}` LIMIT 1000",
    height=120,
    key="mod1_query_input"
)
simulate_large = st.sidebar.checkbox("Simulate large dataset?", key="mod1_simulate")
dup_factor = st.sidebar.number_input(
    "Duplication factor (if simulating)", min_value=1, max_value=200, value=10, key="mod1_dup_factor"
)

if st.button("Run Query Performance Test", key="mod1_run"):
    if not query_input.strip():
        st.error("Please enter a SQL SELECT query.")
    else:
        with engine.connect() as conn:
            target_table = selected_table
            temp_table = f"{selected_table}_sim_tmp"
            if simulate_large and dup_factor > 1:
                st.info(f"Simulating large dataset {dup_factor}x into `{temp_table}`...")
                try:
                    conn.execute(text(f"DROP TABLE IF EXISTS `{temp_table}`"))
                    conn.execute(text(f"CREATE TABLE `{temp_table}` AS SELECT * FROM `{selected_table}`"))
                    for _ in range(dup_factor - 1):
                        conn.execute(text(f"INSERT INTO `{temp_table}` SELECT * FROM `{selected_table}`"))
                    target_table = temp_table
                except Exception as e:
                    st.error(f"Simulation failed: {e}")
                    target_table = selected_table

            safe_query = query_input.replace(selected_table, target_table)
            try:
                tracemalloc.start()
                t0 = time.time()
                df_res = pd.read_sql(safe_query, conn)
                t1 = time.time()
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                st.success(f"Executed in {t1-t0:.4f}s, rows: {df_res.shape[0]}, peak memory: {peak/1024/1024:.2f} MB")
                if df_res.shape[0] > 0:
                    st.dataframe(df_res.head(20))
            except Exception as e:
                st.error(f"Query execution failed: {e}")

            if simulate_large and target_table == temp_table:
                if st.checkbox("Drop temporary simulated table?", value=True, key="mod1_drop_temp"):
                    try:
                        conn.execute(text(f"DROP TABLE IF EXISTS `{temp_table}`"))
                        st.info(f"Temporary table `{temp_table}` dropped.")
                    except Exception as e:
                        st.warning(f"Could not drop temporary table: {e}")

# -------------------------
# Module 2: Indexing Techniques
# -------------------------
st.header("2️⃣ Indexing Techniques Exploration")
index_col = st.text_input("Column to test indexing", value="", key="mod2_index_col")
index_query = st.text_area(f"SQL SELECT query to test", value=f"SELECT * FROM `{selected_table}` WHERE {index_col} IS NOT NULL LIMIT 1000", height=120, key="mod2_query")

if st.button("Run Indexing Test", key="mod2_run"):
    if not index_col.strip() or not index_query.strip():
        st.error("Provide both a column name and a test query.")
    else:
        idx_name = f"idx_{index_col}"
        with engine.begin() as conn:
            try:
                conn.execute(text(f"DROP INDEX {idx_name} ON `{selected_table}`"))
            except Exception:
                pass
            # Without index
            t0 = time.time()
            _ = pd.read_sql(index_query, conn)
            t_no_index = time.time() - t0
            st.write(f"⏱ Execution time **without index**: {t_no_index:.4f}s")
            # With index
            created = False
            try:
                conn.execute(text(f"CREATE INDEX {idx_name} ON `{selected_table}`({index_col})"))
                created = True
                st.success(f"Index `{idx_name}` created on `{selected_table}({index_col})`")
            except Exception as e:
                st.warning(f"Could not create index: {e}")
            if created:
                t0 = time.time()
                _ = pd.read_sql(index_query, conn)
                t_index = time.time() - t0
                st.write(f"⏱ Execution time **with index**: {t_index:.4f}s")
                improvement = ((t_no_index - t_index) / t_no_index) * 100
                st.info(f"✅ Time improvement: {improvement:.2f}%")
                if st.checkbox("Drop index to restore?", value=True, key="mod2_drop"):
                    try:
                        conn.execute(text(f"DROP INDEX {idx_name} ON `{selected_table}`"))
                        st.info(f"Index `{idx_name}` dropped.")
                    except Exception as e:
                        st.warning(f"Failed to drop index: {e}")

# -------------------------
# Module 3 & 4: Query Logs, Feature Exploration, In-DB ML
# -------------------------
st.header("3️⃣ & 4️⃣ Query Logs & Feature Analysis / In-DB ML")
conn_log = sqlite3.connect("query_logs.db")
c_log = conn_log.cursor()
c_log.execute("""
CREATE TABLE IF NOT EXISTS query_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT,
    execution_time REAL,
    rows_returned INTEGER,
    peak_memory_mb REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn_log.commit()

# Query Input
query_perf = st.text_area("Enter SELECT query to test & log", value=f"SELECT * FROM `{selected_table}` LIMIT 1000", height=120, key="mod3_query")

if st.button("Run & Log Query", key="mod3_run"):
    if not query_perf.strip():
        st.error("Enter a query first")
    else:
        with engine.connect() as conn:
            try:
                tracemalloc.start()
                t0 = time.time()
                df_perf = pd.read_sql(query_perf, conn)
                t1 = time.time()
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                elapsed = t1 - t0
                n_rows = len(df_perf)
                peak_mb = peak / 1024 / 1024

                st.success(f"Executed in {elapsed:.4f}s, rows: {n_rows}, peak memory: {peak_mb:.2f} MB")
                if n_rows > 0:
                    st.dataframe(df_perf.head(10))

                c_log.execute(
                    "INSERT INTO query_logs (query, execution_time, rows_returned, peak_memory_mb) VALUES (?, ?, ?, ?)",
                    (query_perf, elapsed, n_rows, peak_mb)
                )
                conn_log.commit()
            except Exception as e:
                st.error(f"Query failed: {e}")

# Show last 50 logs
log_df = pd.read_sql("SELECT * FROM query_logs ORDER BY timestamp DESC LIMIT 50", conn_log)
st.subheader("Recent Queries (last 50)")
st.dataframe(log_df)

if not log_df.empty:
    log_df["query_len"] = log_df["query"].apply(len)
    log_df["num_spaces"] = log_df["query"].apply(lambda x: x.count(" "))
    log_df["num_tables"] = log_df["query"].apply(lambda x: x.upper().count("FROM"))

    st.subheader("Scatter Plot: Query Length vs Execution Time")
    plt.figure(figsize=(6,4))
    plt.scatter(log_df["query_len"], log_df["execution_time"])
    plt.xlabel("Query Length")
    plt.ylabel("Execution Time (s)")
    st.pyplot(plt)

    st.subheader("Bar Chart: Feature vs Execution Time Trend")
    feature_avg = log_df.groupby("num_tables")["execution_time"].mean().reset_index()
    st.bar_chart(feature_avg.set_index("num_tables"))

    # Exploratory models
    X_feat = log_df[["query_len", "num_spaces", "num_tables"]]
    y_time = log_df["execution_time"]
    models_explore = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    for name, model in models_explore.items():
        try:
            model.fit(X_feat, y_time)
            importance = getattr(model, "coef_", None)
            if importance is None:
                importance = getattr(model, "feature_importances_", None)
            if importance is not None:
                fi_df = pd.DataFrame({"Feature": X_feat.columns, "Importance": importance}).sort_values("Importance", ascending=False)
                st.subheader(f"{name} Feature Importance for Execution Time")
                st.dataframe(fi_df)
                st.bar_chart(fi_df.set_index("Feature"))
        except Exception as e:
            st.warning(f"{name} exploratory analysis failed: {e}")

# Optional EXPLAIN plan
st.subheader("Query Plan Visualization (Optional)")
if st.button("Show EXPLAIN Plan for Last Query", key="mod3_explain"):
    if log_df.empty:
        st.info("Run a query first to visualize EXPLAIN plan.")
    else:
        last_query = log_df.iloc[0]["query"]
        try:
            with engine.connect() as conn:
                explain_df = pd.read_sql(f"EXPLAIN {last_query}", conn)
                st.dataframe(explain_df)
        except Exception as e:
            st.warning(f"Could not get EXPLAIN plan: {e}")

# -------------------------
# Automatic Full-DB Index Recommendation
# -------------------------
st.header("💡 One-Click Full Database Index Recommendation")
if st.button("Analyze All Tables for Index Suggestions"):
    recommendations = []
    with engine.connect() as conn:
        for table in table_list:
            try:
                df_sample = pd.read_sql(f"SELECT * FROM `{table}` LIMIT 5000", conn)
                for col in df_sample.columns:
                    if df_sample[col].nunique() / len(df_sample) < 0.8:  # reasonable cardinality
                        recommendations.append({"table": table, "column": col})
            except Exception as e:
                continue
    if recommendations:
        rec_df = pd.DataFrame(recommendations)
        st.subheader("Suggested Indexes for Performance Improvement")
        st.dataframe(rec_df)
    else:
        st.info("No suitable index suggestions found.")
