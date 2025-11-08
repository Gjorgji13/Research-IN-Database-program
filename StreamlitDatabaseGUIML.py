import random
import sqlite3
import time
import tracemalloc
from contextlib import contextmanager
from typing import List, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, inspect

from indatabase_ai_ml import selected_table

# -------------------------
# Page config & UI Setup
# -------------------------
st.set_page_config(page_title="Database Research Lab", layout="wide")


# -------------------------
# Custom CSS for Professional Look
# -------------------------
def apply_custom_css():
    st.markdown("""
        <style>
            /* General Streamlit tweaks */
            .main { padding-top: 1rem; }
            /* Style for the tabs */
            .stTabs [data-baseweb="tab-list"] {
                gap: 15px;
                padding-bottom: 10px;
                border-bottom: 2px solid #2980b9; /* Darker blue line */
            }
            .stTabs [data-baseweb="tab"] {
                height: 50px;
                white-space: nowrap;
                background-color: transparent;
                border-radius: 4px 4px 0 0;
                padding: 10px 20px;
                font-weight: bold;
                color: #A0A0A0;
                transition: color 0.3s;
            }
            .stTabs [aria-selected="true"] {
                color: #2980b9; /* Active tab color */
                border-top: 3px solid #2980b9; /* Accent color */
                border-left: 1px solid #E6EAF1;
                border-right: 1px solid #E6EAF1;
                border-bottom: 1px solid #F0F2F6;
            }
            /* Make primary buttons look sharp */
            div.stButton > button:first-child {
                border-color: #2980b9;
                background-color: #2980b9;
                color: white;
                font-weight: bold;
                padding: 0.5rem 1rem;
                transition: background-color 0.3s;
            }
            div.stButton > button:first-child:hover {
                border-color: #2c3e50;
                background-color: #2c3e50;
            }
            /* Custom styling for the AI Query Analyzer suggestion section */
            .ai-suggestion {
                border: 1px solid #f39c12;
                border-radius: 5px;
                padding: 10px;
                background-color: #fffde7;
            }
        </style>
    """, unsafe_allow_html=True)


apply_custom_css()

st.title("🧪 Database Research Lab — Professional SQL GUI")
st.markdown("---")

# -------------------------
# Session State Initialization & Configuration
# -------------------------
if "engine" not in st.session_state:
    st.session_state["engine"] = None
if "connected" not in st.session_state:
    st.session_state["connected"] = False
if "selected_db_name" not in st.session_state:
    st.session_state["selected_db_name"] = None
if "tables" not in st.session_state:
    st.session_state["tables"] = []
if "selected_table" not in st.session_state:
    st.session_state["selected_table"] = "(no tables found)"
if "ai_q_text" not in st.session_state:
    st.session_state["ai_q_text"] = "SELECT * FROM items LIMIT 10;"
if "qb_filters" not in st.session_state:
    # Structure: [(key, column, operator, value)]
    st.session_state["qb_filters"] = []
if "qb_join" not in st.session_state:
    # Structure: (join_type, other_table, main_col, other_col)
    st.session_state["qb_join"] = None
if "kv_store" not in st.session_state:  # New
    st.session_state["kv_store"] = {}  # Simple in-memory KV store simulation
if "ai_suggested_query_ran" not in st.session_state:
    st.session_state["ai_suggested_query_ran"] = False


# -------------------------
# Helpers & small utils (Refined and consolidated)
# -------------------------
@contextmanager
def sqlite_conn(path):
    conn = sqlite3.connect(path, check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()


def norm_query_for_hash(q: str) -> str:
    return hashlib.md5(q.strip().encode("utf-8")).hexdigest()


def ensure_perf_db(path="perf_logs.db"):
    with sqlite_conn(path) as c:
        cur = c.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS perf_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                db_type TEXT,
                db_name TEXT,
                query_hash TEXT,
                query_text TEXT,
                rows_returned INTEGER,
                exec_time REAL,
                peak_memory_mb REAL,
                indexed INTEGER
            )
        """)
        c.commit()


ensure_perf_db()  # Ensure log DB is ready


def log_performance_db(db_type, db_name, query, rows, exec_time, peak_mb, indexed=0, path="perf_logs.db"):
    qh = norm_query_for_hash(query)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with sqlite_conn(path) as c:
        cur = c.cursor()
        cur.execute(
            "INSERT INTO perf_log (timestamp, db_type, db_name, query_hash, query_text, rows_returned, exec_time, peak_memory_mb, indexed) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, db_type, db_name, qh, query, rows, exec_time, peak_mb, int(indexed))
        )
        c.commit()


def read_perf_logs(path="perf_logs.db", limit=500):
    with sqlite_conn(path) as c:
        df = pd.read_sql_query(f"SELECT * FROM perf_log ORDER BY timestamp DESC LIMIT {limit}", c)
    return df


def safe_sql_identifier(name: str) -> str:
    """Safe quotes identifiers for MySQL/SQLite."""
    return f"`{name.replace('`', '')}`"


def get_table_metadata(engine, table_name):
    """Retrieves column names and types for a given table."""
    try:
        inspector = inspect(engine)
        cols = inspector.get_columns(table_name)
        return {c['name']: str(c['type']) for c in cols}
    except Exception:
        return {}


def execute_and_measure(query: str, engine, fetch_sample=1000):
    """Execute query and measure time and memory usage."""
    tracemalloc.start()
    t0 = time.time()
    df = pd.DataFrame()
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn)
    except Exception as e:
        raise e
    finally:
        t1 = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    elapsed = t1 - t0
    peak_mb = peak / 1024 / 1024
    return df, elapsed, peak_mb

# # --------------------------------------------------
# # INDEX HELPER FUNCTIONS (FIX for UnboundLocalError)
# # --------------------------------------------------

    # -------------------------
    # 2. Indexing Experiments
    # -------------------------
    st.markdown("---")
    st.markdown("## 2. Indexing Techniques & Tester")
    st.write("Compare query performance before and after creating a temporary index.")
    st.info("""
        **How to Test:** Enter the column name and a test query (which **must** filter on that column, e.g., `WHERE price > 100`). The app will run the query:
        1. **No Index** (Baseline)
        2. **Index Created** (The performance should improve)
        3. **Index Dropped** (Cleanup)
    """)

    # --- AI Suggestion & Session State Management ---
    initial_index_value = st.session_state.get("index_col_input", "")

    if "_ai_suggested_index_col" in st.session_state:
        # 1. Overwrite with the AI value and clean it immediately
        initial_index_value = clean_index_column(st.session_state.pop("_ai_suggested_index_col"))
        # 2. Force set the clean value into the session state for the widget
        st.session_state["index_col_input"] = initial_index_value

    # --- User Input Field ---
    index_column_raw = st.text_input(
        "Column to index (exact name)",
        value=initial_index_value,
        key="index_col_input"
    )

    # 3. Apply final cleanup to the input value before use
    index_column = clean_index_column(index_column_raw)

    # --- Safety Check: Column Existence and Persistence (Rerun Logic) ---

    # Check if the cleaned value differs from the raw input (means we fixed a syntax issue like 'price >')
    if index_column and index_column != index_column_raw:
        st.info("Input cleaned of invalid characters. Rerunning to display corrected value.")
        st.session_state["index_col_input"] = index_column
        st.experimental_rerun()

    # Check for column existence (fixes the 'quantity' error)
    safe_fallback_col = "id"  # Default fallback
    if index_column:
        try:
            col_metadata = get_table_metadata(engine, selected_table)
            safe_fallback_col = next(iter(col_metadata.keys()), 'id')  # Find a true safe column

            if index_column not in col_metadata:
                st.warning(
                    f"⚠️ Column '{index_column}' does not exist in table '{selected_table}'. Falling back to '{safe_fallback_col}'.")

                # Update the variable and session state to the safe column and rerun
                index_column = safe_fallback_col
                if index_column != st.session_state.get("index_col_input"):
                    st.session_state["index_col_input"] = index_column
                    st.experimental_rerun()

        except Exception as e:
            st.error(f"Failed to fetch table metadata for column check: {e}")

    # Use the cleaned/safe column name for the test query default
    index_test_query = st.text_area(
        "Test query (MUST use the column in WHERE)",
        value=f"SELECT * FROM {safe_sql_identifier(selected_table)} WHERE {safe_sql_identifier(index_column)} IS NOT NULL LIMIT 1000",
        height=80,
        key="index_test_query_area"
    )

    if st.button("Run Indexing Comparison (No-Index -> Index -> Drop Index)", key="run_index_test_btn"):
        if not index_column or not index_test_query.strip():
            st.error("Provide a valid column name and a test query.")
            return

        idx_name = f"idx_test_{index_column}_{hashlib.sha1(os.urandom(10)).hexdigest()[:5]}"
        results = []  # Initialize results list here

        # --- 1. No Index Run ---
        st.info("Phase 1/3: Running query **without** index...")
        try:
            df_no, t_no, m_no = execute_and_measure(index_test_query, engine)
            results.append({"Run": "No Index", "Time (s)": t_no})
            log_performance_db(db_mode, st.session_state["selected_db_name"], index_test_query, len(df_no), t_no,
                               m_no, indexed=0)
            st.success(f"No Index Time: {t_no:.4f}s")
        except Exception as e:
            st.error(f"No Index Run Failed: {e}")
            return

        # --- 2. Create Index ---
        st.info(f"Phase 2/3: Creating index `{idx_name}` on column `{index_column}`...")
        try:
            create_index(engine, selected_table, index_column, idx_name)  # Uses the CLEANED index_column
            st.success("Index created.")
        except Exception as e:
            st.error(f"Index Creation Failed: {e}. Aborting test.")
            # Important: Attempt to drop the index in case of partial creation
            try:
                drop_index(engine, selected_table, idx_name, db_mode)
            except:
                pass
            return

        # --- 3. Index Run ---
        st.info(f"Running query **with** index `{idx_name}`...")
        try:
            df_idx, t_idx, m_idx = execute_and_measure(index_test_query, engine)
            results.append({"Run": "With Index", "Time (s)": t_idx})
            log_performance_db(db_mode, st.session_state["selected_db_name"], index_test_query, len(df_idx), t_idx,
                               m_idx, indexed=1)
            st.success(f"With Index Time: {t_idx:.4f}s")
        except Exception as e:
            st.error(f"Indexed Run Failed: {e}")
            t_idx = t_no

        # --- 4. Drop Index (Cleanup) ---
        st.info("Phase 3/3: Dropping index (Cleanup)...")
        try:
            drop_index(engine, selected_table, idx_name, db_mode)
            st.success("Index dropped successfully. Environment restored.")
        except Exception as e:
            st.error(f"Index Drop Failed: {e}. Please drop index `{idx_name}` manually.")

        # --- Results Visualization ---
        df_r = pd.DataFrame(results).set_index("Run")
        st.subheader("Indexing Comparison Results")
        st.dataframe(df_r)

        fig, ax = plt.subplots(figsize=(6, 3))
        df_r["Time (s)"].plot(kind='bar', ax=ax, color=['#e74c3c', '#27ae60'])
        ax.set_title("Query Time: No Index vs. With Index")
        ax.set_ylabel("Execution Time (s)")
        ax.tick_params(axis='x', rotation=0)
        st.pyplot(fig)

        st.markdown(
            f"""
            > **Research Insight:** The performance difference, typically measured by the ratio of **No Index Time / With Index Time**, demonstrates the efficiency gain 
            > of avoiding a full table scan and using an optimized index structure (usually a B-Tree) to quickly locate data.
            """
        )

# -------------------------
# AI & ML Functions (Refined)
def train_ai_from_logs(limit=1000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Analyzes logs to find frequent filter columns (basic in-DB ML)."""
    df_logs = read_perf_logs(limit=limit)
    if df_logs.empty: return pd.DataFrame(), pd.DataFrame()

    tokens = []
    for q in df_logs["query_text"].astype(str):
        uq = q.upper()
        if "WHERE" in uq:
            try:
                where_part = uq.split("WHERE", 1)[1]
                for delim in ["LIMIT", "ORDER BY", "GROUP BY", ";"]:
                    if delim in where_part: where_part = where_part.split(delim, 1)[0]
                conds = [c.strip() for c in where_part.replace("AND", "|").replace("OR", "|").split("|")]
                for c in conds:
                    for op in ["=", "<", ">", " LIKE ", " IN ", " IS "]:
                        if op in c:
                            token = c.split(op, 1)[0].strip().strip("()`[]")
                            if token: tokens.append(token.lower())
                            break
            except Exception:
                continue

    if not tokens:
        freq_df = pd.DataFrame()
    else:
        s = pd.Series(tokens).value_counts()
        freq_df = pd.DataFrame({"column": s.index, "count": s.values})

    perf_summary = df_logs.groupby("query_hash").agg(
        avg_exec_time=("exec_time", "mean"),
        avg_peak_mb=("peak_memory_mb", "mean"),
        runs=("id", "count")
    ).reset_index()

    if not freq_df.empty:
        st.session_state["ai_learned_col"] = freq_df.iloc[0]["column"]

    return freq_df, perf_summary


def get_ai_filter_suggestion(df: pd.DataFrame) -> str:
    """Provides an AI filter suggestion based on learned column or random sampling."""
    top_col = st.session_state.get("ai_learned_col")

    if top_col and top_col in df.columns:
        col = top_col
    else:
        cols = df.columns.tolist()
        if not cols: return ""
        col = random.choice(cols)

    if pd.api.types.is_numeric_dtype(df[col]):
        mean_val = df[col].mean()
        return f"{safe_sql_identifier(col)} > {mean_val:.2f}"
    elif pd.api.types.is_object_dtype(df[col]):
        sample_value = df[col].dropna().mode()
        if not sample_value.empty:
            val = str(sample_value[0]).replace("'", "''")  # Escape single quote for SQL
            return f"{safe_sql_identifier(col)} = '{val}'"
    return ""


def suggest_ai_sql(engine, table, columns: List[str], user_selects: List[str] = None) -> str:
    """Generates a valid, sensible SQL query based on table/column hints."""
    if not columns: return f"SELECT * FROM {table} LIMIT 10;"

    # 1. Selects (Use user_selects if provided, otherwise pick a few)
    select_candidates = user_selects or columns[:3] if len(columns) > 3 else columns

    # Filter out potential primary keys/IDs unless explicitly requested for simple aggregation
    select_candidates_filtered = [c for c in select_candidates if 'id' not in c.lower() or c in select_candidates]

    # 2. Heuristics for Group By / Aggregation
    num_candidates = [c for c in columns if
                      any(k in c.lower() for k in ("amount", "price", "count", "total", "score", "num", "qty"))]
    text_candidates = [c for c in columns if
                       any(k in c.lower() for k in ("name", "type", "category", "city", "region", "status"))]

    if num_candidates and text_candidates:
        num_col = safe_sql_identifier(num_candidates[0])
        text_col = safe_sql_identifier(text_candidates[0])

        # Find all descriptive, non-numeric columns to use for grouping
        # For simplicity and correctness with ONLY_FULL_GROUP_BY, we group by ALL selected non-aggregated, non-ID columns.
        group_candidates = [c for c in select_candidates_filtered if c in text_candidates]

        # If no clear text candidates in selected, just use the first few selected columns
        if not group_candidates:
            group_candidates = select_candidates_filtered[:2]

        # Ensure all columns intended for GROUP BY are safe
        group_by_cols_list = [safe_sql_identifier(c) for c in group_candidates]
        group_by_cols = ", ".join(group_by_cols_list)

        # Build the final SELECT list: all group columns + aggregated columns
        selects = group_by_cols_list
        selects.append("COUNT(*) AS record_count")
        selects.append(f"AVG({num_col}) AS avg_{num_candidates[0]}")

        # Ensure the GROUP BY list is not empty
        if not group_by_cols:
            return build_sql(selects=["*"], table=safe_sql_identifier(table), limit=20)

        return build_sql(
            selects=selects,
            table=safe_sql_identifier(table),
            group=group_by_cols,
            order="record_count",
            order_dir="DESC",
            limit=10
        )
    else:
        # Simple query with limit
        selects = [safe_sql_identifier(c) for c in select_candidates]
        return build_sql(selects=selects, table=safe_sql_identifier(table), limit=20)


# --------------------------------------------------
# Core SQL Builder Logic
# --------------------------------------------------

def build_sql(selects, table, joins=None, where=None, group=None, having=None, order=None, order_dir=None, limit=None):
    """Assembles SQL from modular parts."""
    sel = ", ".join([s.strip() for s in selects if s is not None and str(s).strip() != ""])
    if not sel: sel = "*"

    query = f"SELECT {sel} FROM {table}"

    if joins: query += f" {joins}"
    if where: query += f" WHERE {where}"
    if group: query += f" GROUP BY {group}"
    if having: query += f" HAVING {having}"
    if order and order != "(none)":
        order_col = safe_sql_identifier(order)
        query += f" ORDER BY {order_col} {order_dir}"
    if limit: query += f" LIMIT {limit}"

    return query


def render_dynamic_where_builder(table_metadata):
    """Renders the UI for building WHERE clauses."""
    st.subheader("Filter Conditions (WHERE)")

    # Button to add new filter row
    if st.button("➕ Add Filter Condition", key="add_filter_btn"):
        # Use a unique key for each new filter
        new_key = time.time()
        st.session_state["qb_filters"].append((new_key, None, '=', ''))

    # Display current filters
    remove_keys = []
    for i, (key, col, op, val) in enumerate(st.session_state["qb_filters"]):
        filter_col1, filter_col2, filter_col3, filter_col4, filter_col5 = st.columns([1, 1.5, 1, 3, 0.5])

        with filter_col1:
            st.write(f"Condition {i + 1}:")

        with filter_col2:
            current_col = filter_col2.selectbox(
                "Column",
                options=["(select)"] + list(table_metadata.keys()),
                index=(list(table_metadata.keys()).index(col) + 1 if col in table_metadata else 0),
                key=f"col_{key}",
                label_visibility="collapsed"
            )
        with filter_col3:
            current_op = filter_col3.selectbox(
                "Operator",
                options=["=", ">", "<", ">=", "<=", "<>", "LIKE"],
                index=["=", ">", "<", ">=", "<=", "<>", "LIKE"].index(op) if op in ["=", ">", "<", ">=", "<=", "<>",
                                                                                    "LIKE"] else 0,
                key=f"op_{key}",
                label_visibility="collapsed"
            )
        with filter_col4:
            current_val = filter_col4.text_input(
                "Value",
                value=val,
                key=f"val_{key}",
                label_visibility="collapsed"
            )
        with filter_col5:
            if filter_col5.button("🗑️", key=f"del_{key}"):
                remove_keys.append(key)

        # Update session state after user interaction
        st.session_state["qb_filters"][i] = (key, current_col, current_op, current_val)

    # Process removals
    st.session_state["qb_filters"] = [f for f in st.session_state["qb_filters"] if f[0] not in remove_keys]

    # Convert filters to SQL WHERE string
    where_parts = []
    # Pre-calculate the replacement character to avoid backslash inside f-string expression
    QUOTE_ESCAPE = "'"

    for key, col, op, val in st.session_state["qb_filters"]:
        if col and col != "(select)" and val.strip():
            # Check if value needs quotes (based on simple heuristic)
            if op == "LIKE":
                # Escape single quotes using double quotes or a standard method
                escaped_val = val.strip().replace(QUOTE_ESCAPE, '\"')
                val_clean = f"'%{escaped_val}%'"
            elif not val.strip().replace('.', '', 1).isdigit() and op in ("=", "<>", "LIKE"):
                # Escape single quotes and wrap in quotes for text comparison
                escaped_val = val.strip().replace(QUOTE_ESCAPE, '\"')
                val_clean = f"'{escaped_val}'"
            else:
                # Numeric value
                val_clean = val.strip()

            where_parts.append(f"{safe_sql_identifier(col)} {op} {val_clean}")

    return " AND ".join(where_parts)


# --------------------------------------------------
# UI Modules (Mapped to Tabs)
# --------------------------------------------------

def render_connection_and_setup():
    """Renders the Connection/Setup tab."""
    st.header("⚙️ Database Connection & Setup")

    # -------------------------
    # TUTORIAL TEXT START
    # -------------------------
    st.markdown("""
        **👋 Welcome to the Database Research Lab!**
        To begin your experiments in the other tabs, you must first connect to a database and select a table.

        **Follow these steps:**
        1. **Select Engine (Sidebar):** Choose between **SQLite** (file-based, recommended for quick tests) or **MySQL** (for a real server environment).
        2. **Connect:** Enter the required details in the sidebar and click the connection button.
        3. **Select Table:** After connecting, choose your target table from the sidebar dropdown.
    """)
    # -------------------------
    # TUTORIAL TEXT END
    # -------------------------

    # 1. Sidebar: DB engine mode selection
    db_mode = st.sidebar.selectbox("Select DB engine", ["SQLite (file)", "MySQL"], index=0, key="db_mode_choice")

    # Initialize connection variables
    engine, connected, selected_db_name, tables = None, False, None, []
    sqlite_file, mysql_user, mysql_pass, mysql_host, mysql_port = "research_sample.db", "root", "", "localhost", "3306"

    # 2. Connection inputs & logic
    with st.sidebar.container(border=True):
        st.subheader("Connection Details")
        if db_mode == "SQLite (file)":
            sqlite_file = st.text_input("SQLite file path", value=sqlite_file, key="sqlite_path")
            if st.button("Test SQLite Connection", key="test_sqlite"):
                try:
                    selected_db_name = os.path.abspath(sqlite_file)
                    engine = create_engine(f"sqlite:///{selected_db_name}")
                    connected = os.path.exists(sqlite_file)
                    if connected:
                        st.sidebar.success(f"SQLite connected: {selected_db_name}")
                        st.session_state["engine"] = engine
                        st.session_state["connected"] = True
                        st.session_state["selected_db_name"] = selected_db_name
                    else:
                        st.sidebar.warning(f"SQLite file not found: {sqlite_file}. Ready to create sample data.")
                        st.session_state["engine"] = engine  # Engine created even if file is new
                        st.session_state["connected"] = False
                except Exception as e:
                    st.sidebar.error(f"SQLite error: {e}")
                    st.session_state["connected"] = False
        else:
            mysql_host = st.text_input("MySQL host", value=mysql_host, key="mysql_host")
            mysql_port = st.text_input("MySQL port", value=mysql_port, key="mysql_port")
            mysql_user = st.text_input("MySQL user", value=mysql_user, key="mysql_user")
            mysql_pass = st.text_input("MySQL password", value=mysql_pass, type="password", key="mysql_pass")
            mysql_url_base = f"mysql+mysqlconnector://{mysql_user}:{mysql_pass}@{mysql_host}:{mysql_port}/"

            if st.button("Test MySQL Connection", key="test_mysql"):
                try:
                    engine_tmp = create_engine(mysql_url_base)
                    with engine_tmp.connect() as tmpc:
                        databases = [r[0] for r in tmpc.execute(text("SHOW DATABASES")).fetchall()]
                        databases = [d for d in databases if
                                     d.lower() not in ("information_schema", "mysql", "performance_schema", "sys")]

                    st.session_state["mysql_databases"] = databases
                    st.sidebar.success(f"Successfully connected to MySQL server. Choose database below.")

                except Exception as e:
                    st.sidebar.error(f"MySQL connection failed: {e}")
                    st.session_state["connected"] = False

            if "mysql_databases" in st.session_state and st.session_state["mysql_databases"]:
                selected_db_name = st.selectbox("Choose database", st.session_state["mysql_databases"],
                                                key="mysql_db_select")
                if st.button("Connect to Database", key="connect_db"):
                    DB_URL = f"{mysql_url_base}{selected_db_name}"
                    engine = create_engine(DB_URL)
                    st.session_state["engine"] = engine
                    st.session_state["connected"] = True
                    st.session_state["selected_db_name"] = selected_db_name
                    st.sidebar.success(f"MySQL connected: {selected_db_name}")

    # 3. Table list/selection (Updated after connection)
    if st.session_state["connected"] and st.session_state["engine"]:
        try:
            inspector = inspect(st.session_state["engine"])
            tables = inspector.get_table_names()
            st.session_state["tables"] = tables
        except Exception as e:
            st.warning(f"Could not retrieve tables: {e}")
            st.session_state["tables"] = ["(no tables found)"]

        if st.session_state["tables"]:
            selected_table = st.sidebar.selectbox("Select table for experiments", st.session_state["tables"],
                                                  key="sel_table")
            st.session_state["selected_table"] = selected_table

    # 4. Sample Data Generator (Conditional on SQLite)
    st.subheader("Sample Data Generator (SQLite only)")

    # -------------------------
    # TUTORIAL TEXT START
    # -------------------------
    st.info("""
        **💡 Tip:** Use this feature if you don't have existing data. The `items` and `sales` tables are designed to 
        test joins, filters, and indexing experiments in the **Research Lab**. After creating, **re-test the connection** in the sidebar to refresh the table list.
    """)
    # -------------------------
    # TUTORIAL TEXT END
    # -------------------------

    if db_mode.startswith("SQLite"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Create sample table `items` (1k rows)"):
                engine = st.session_state["engine"]
                df_sample = pd.DataFrame({
                    "id": range(1, 1001),
                    "category": np.random.choice(["A", "B", "C", "D"], size=1000),
                    "price": np.random.randint(5, 1000, size=1000),
                    "region": np.random.choice(["north", "south", "east", "west"], size=1000),
                    "created_at": pd.date_range("2020-01-01", periods=1000, freq="H")
                })
                df_sample.to_sql("items", engine, if_exists="replace", index=False)
                st.success(
                    "Sample table `items` created. **Please re-test connection in sidebar to refresh table list.**")
        with col2:
            if st.button("Create sample table `sales` (5k rows)"):
                engine = st.session_state["engine"]
                df_sample = pd.DataFrame({
                    "id": range(1, 5001),
                    "item_id": np.random.randint(1, 1001, size=5000),
                    "qty": np.random.randint(1, 10, size=5000),
                    "amount": np.random.random(5000) * 200,
                    "sale_date": pd.date_range("2020-01-01", periods=5000, freq="H")
                })
                df_sample.to_sql("sales", engine, if_exists="replace", index=False)
                st.success(
                    "Sample table `sales` created. **Please re-test connection in sidebar to refresh table list.**")
    else:
        st.info("Sample data creation is only available for SQLite.")


def render_dynamic_sql_builder():
    """Renders the Dynamic SQL Builder (No-Code SQL) tab."""
    st.header("🚀 Dynamic SQL Builder — No-Code Data Explorer")

    # -------------------------
    # TUTORIAL TEXT START
    # -------------------------
    st.markdown("""
        This tool helps you construct complex SQL queries using simple inputs, without writing any code.
        It's great for:
        * **Learning:** See how SQL clauses (SELECT, WHERE, JOIN, GROUP BY) are assembled.
        * **Preparation:** Quickly build a query to run performance tests on in the **Research Lab**.
        * **Discovery:** Find interesting aggregations or filter conditions in your data.
    """)
    # -------------------------
    # TUTORIAL TEXT END
    # -------------------------

    if not st.session_state["connected"] or st.session_state["selected_table"] == "(no tables found)":
        st.warning("Please connect to a database and select a table in the **Connection & Setup** tab first.")
        return

    engine = st.session_state["engine"]
    qb_table = st.session_state["selected_table"]
    col_metadata = get_table_metadata(engine, qb_table)
    col_names = list(col_metadata.keys())

    # --- 1. Table Preview
    st.subheader(f"📋 Working Table: `{qb_table}`")
    try:
        with st.expander("Show Table Preview (First 10 Rows)"):
            with engine.connect() as conn:
                preview_df = pd.read_sql(f"SELECT * FROM {safe_sql_identifier(qb_table)} LIMIT 10", conn)
                st.dataframe(preview_df)
    except Exception as e:
        st.warning(f"Preview not available: {e}")

    # --- 2. SELECT Columns
    st.markdown("### 1️⃣ SELECT: Choose Columns")
    st.info(
        "Select the columns you want to view or aggregate. Leaving this empty or selecting none will default to `SELECT *`.")
    selected_cols = st.multiselect("Columns to SELECT", col_names, default=col_names[:3] if col_names else [],
                                   key="qb_select_cols")
    if not selected_cols: selected_cols = col_names
    select_sql = [safe_sql_identifier(c) for c in selected_cols]

    # --- 3. JOIN Builder
    st.markdown("### 2️⃣ JOIN: Connect Tables (Optional)")
    st.info(
        "Build an `INNER` or `LEFT` join to combine data from another table based on matching column values (e.g., `item_id` in both tables).")
    all_tables = [t for t in st.session_state["tables"] if t != qb_table]
    join_type, join_table, join_on_main, join_on_other = st.session_state["qb_join"] or (None, None, None, None)
    join_clause = None

    join_col1, join_col2, join_col3, join_col4 = st.columns(4)
    with join_col1:
        join_type = st.selectbox("Type", ["(none)", "INNER JOIN", "LEFT JOIN"], key="qb_join_type")
    if join_type != "(none)" and all_tables:
        with join_col2:
            join_table = st.selectbox("Table", all_tables, key="qb_join_table")
        with join_col3:
            join_on_main = st.selectbox(f"On {qb_table} Col", col_names, key="qb_join_maincol")
        with join_col4:
            try:
                other_col_names = list(get_table_metadata(engine, join_table).keys()) if join_table else []
                join_on_other = st.selectbox(f"On {join_table} Col", other_col_names, key="qb_join_othercol")
            except Exception:  # Handle case where join_table is un-inspectable
                join_on_other = st.text_input(f"On {join_table} Col (manual)", key="qb_join_othercol_manual")

        if join_table and join_on_main and join_on_other:
            join_clause = f"{join_type} {safe_sql_identifier(join_table)} ON {safe_sql_identifier(qb_table)}.{safe_sql_identifier(join_on_main)} = {safe_sql_identifier(join_table)}.{safe_sql_identifier(join_on_other)}"
            st.session_state["qb_join"] = (join_type, join_table, join_on_main, join_on_other)
            st.info(f"Join Clause: {join_clause}")
        else:
            st.session_state["qb_join"] = None
    else:
        st.session_state["qb_join"] = None

    # --- 4. WHERE Builder
    st.markdown("### 3️⃣ WHERE: Filter Rows")
    st.info(
        "Use the filter builder to narrow down your results. Each condition is combined with an **AND** operator. Example: `price > 100`.")
    where_sql = render_dynamic_where_builder(col_metadata)

    # --- 5. GROUP BY / HAVING
    st.markdown("### 4️⃣ GROUP BY & HAVING (Optional Aggregation)")
    st.info(
        "Select one or more columns to group by. Use `HAVING` (e.g., `COUNT(*) > 10`) to filter the results *after* aggregation (only groups that satisfy the condition are returned).")
    group_cols_sql = None
    group_cols = st.multiselect("GROUP BY columns", col_names, key="qb_group_cols")
    if group_cols:
        group_cols_sql = ", ".join([safe_sql_identifier(c) for c in group_cols])
        having_exp = st.text_input("HAVING condition (SQL syntax, e.g. COUNT(*) > 10)", key="qb_having_exp")
    else:
        having_exp = None
    having_sql = having_exp.strip() if having_exp else None

    # --- 6. ORDER BY / LIMIT
    st.markdown("### 5️⃣ ORDER & LIMIT")
    col_o1, col_o2, col_o3 = st.columns([2, 1, 1])
    with col_o1:
        order_col = st.selectbox("ORDER BY column", ["(none)"] + col_names, key="qb_order_col")
    with col_o2:
        order_dir = st.radio("Direction", ["ASC", "DESC"], horizontal=True, key="qb_order_dir")
    with col_o3:
        limit_val = st.number_input("LIMIT rows", min_value=1, max_value=5000, value=100, key="qb_limit_val")

    # --- FINAL SQL & Execution
    st.markdown("---")
    st.subheader("Final Generated Query")

    final_query_sql = build_sql(
        selects=select_sql,
        table=safe_sql_identifier(qb_table),
        joins=join_clause,
        where=where_sql or None,
        group=group_cols_sql,
        having=having_sql,
        order=order_col,
        order_dir=order_dir,
        limit=limit_val
    )

    st.code(final_query_sql, language="sql")

    col_run1, col_run2 = st.columns(2)
    with col_run1:
        if st.button("▶️ Execute Generated Query", key="qb_run_btn"):
            try:
                df, t, m = execute_and_measure(final_query_sql, engine)
                st.session_state["qb_result_df"] = df
                st.session_state["qb_result_time"] = t
                st.session_state["qb_result_mem"] = m
                log_performance_db(st.session_state["db_mode_choice"], st.session_state["selected_db_name"],
                                   final_query_sql, len(df), t, m, indexed=0)
            except Exception as e:
                st.error(f"❌ Query execution failed: {e}")
                st.session_state["qb_result_df"] = pd.DataFrame()

    with col_run2:
        if st.button("💡 Get AI Query Suggestion", key="ai_suggest_run_btn"):
            ai_q = suggest_ai_sql(engine, qb_table, col_names, selected_cols)
            st.session_state["ai_q_text"] = ai_q
            st.session_state["ai_suggested_query_ran"] = True

            st.rerun()

    # --- FIX: Immediate AI Suggestion Display (Start) ---
    if st.session_state.get("ai_suggested_query_ran", False):
        st.markdown('<div class="ai-suggestion">', unsafe_allow_html=True)
        st.subheader("💡 Generated AI Query Suggestion")
        st.write(
            "The suggestion has been generated and pre-filled in the **🧠 AI Analyzer & Logs** tab for execution. You can copy the code below to run it immediately.")
        st.code(st.session_state.get("ai_q_text", "Could not generate a suggestion."), language="sql")
        st.markdown('</div>', unsafe_allow_html=True)

        # Reset flag immediately after display to prevent showing on every subsequent rerun
        st.session_state["ai_suggested_query_ran"] = False
    # --- FIX: Immediate AI Suggestion Display (End) ---

    # --- Results Display
    if "qb_result_df" in st.session_state and not st.session_state["qb_result_df"].empty:
        st.subheader("Query Result")
        st.success(
            f"✅ Query executed in **{st.session_state['qb_result_time']:.4f}s** | Peak memory: **{st.session_state['qb_result_mem']:.2f} MB** | Rows: **{len(st.session_state['qb_result_df'])}**")
        st.dataframe(st.session_state["qb_result_df"])


def render_ai_analyzer_and_logs():
    """Renders the AI Analyzer and Logs tab."""
    st.header("🧠 AI Analyzer & Logs — Optimization Insights")

    # -------------------------
    # TUTORIAL TEXT START
    # -------------------------
    st.markdown("""
        This tab provides insights into past database activity, learning how to suggest future optimizations.

        * **Query Logs:** Every query run in the **Dynamic SQL Builder** or **Research Lab** is logged here.
        * **AI Suggestion:** The analyzer inspects your past queries to find frequently filtered columns, suggesting which ones should be indexed to improve performance.
    """)
    # -------------------------
    # TUTORIAL TEXT END
    # -------------------------

    if not st.session_state["connected"]:
        st.warning("Please connect to a database in the **Connection & Setup** tab first.")
        return

    engine = st.session_state["engine"]

    # -------------------------
    # 1. AI Suggestion
    # -------------------------
    st.markdown("## 1. AI Index Suggestion (Learned from History)")

    with st.container(border=True):
        freq_df, perf_summary = train_ai_from_logs(limit=1000)

        if not freq_df.empty:
            st.subheader("Top 5 Most Frequently Filtered Columns (Index Candidates)")

            # Show the data used for the suggestion
            st.dataframe(freq_df.head(5).style.background_gradient(subset=['count'], cmap='YlOrRd'),
                         use_container_width=True)

            # Use the top column for the suggestion
            top_col = freq_df.iloc[0]["column"]
            st.markdown(f"""
                **🤖 AI Recommendation:** Based on your query history, the column **`{top_col}`** is used in filter (WHERE) clauses most often. 
                Creating an index on this column is the most likely way to improve overall query execution time.
            """)

            # Temporary storage to pre-fill the Research Lab
            st.session_state["_ai_suggested_index_col"] = top_col
            st.info("The suggested column is now pre-filled in the **Research Lab** tab for immediate testing.")
        else:
            st.info(
                "No query history yet — run some queries in the **Dynamic SQL Builder** or **Research Lab** tabs first.")

    # -------------------------
    # 2. Historical Logs & Trends
    # -------------------------
    st.markdown("## 2. Historical Query Trends")
    df_logs = read_perf_logs(limit=500)

    if df_logs.empty:
        st.info("No historical logs to display.")
        return

    st.write("Select a query to view its performance trend over time and compare indexed vs. non-indexed runs.")

    # Unique queries for selection
    queries_unique = df_logs["query_text"].unique().tolist()
    sel_q = st.selectbox("Select Logged Query", queries_unique, key="trend_select")

    df_sel = df_logs[df_logs["query_text"] == sel_q].sort_values("timestamp")

    if not df_sel.empty:
        df_chart = df_sel.copy()
        df_chart["timestamp"] = pd.to_datetime(df_chart["timestamp"])
        df_chart["indexed_status"] = df_chart["indexed"].apply(lambda x: "Indexed" if x == 1 else "No Index")

        st.subheader("Execution Time Trend")
        # Separate indexed and non-indexed runs for a clear visual comparison
        fig, ax = plt.subplots(figsize=(10, 4))
        for status, group in df_chart.groupby("indexed_status"):
            ax.plot(group["timestamp"], group["exec_time"], marker='o', label=status)
        ax.set_xlabel("Time of Run")
        ax.set_ylabel("Execution Time (s)")
        ax.set_title("Performance Trend by Index Status")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Raw Performance Data (Last 20 Runs)")
        st.dataframe(df_sel[["timestamp", "exec_time", "peak_memory_mb", "rows_returned", "indexed"]].tail(20))

# --------------------------------------------------
# CRITICAL FIX: INDEX HELPER FUNCTIONS (GLOBAL SCOPE)
# Place these outside and before def render_research_lab()
# --------------------------------------------------


# --- IMPORTANT: Ensure these helper functions are in the global scope ---

# Assumes safe_sql_identifier and text are available from other parts of your script.

def clean_index_column(col_name: str) -> str:
    """Cleans column name to remove operators and quotes, preventing SQL syntax errors."""
    if not col_name:
        return ""
    col_name = col_name.replace('`', '').replace('"', '').replace("'", '').strip()
    col_name = re.sub(r'[\s<=>!]+$', '', col_name).strip()
    return col_name


def create_index(engine, table, col, idx_name):
    """Creates an index on the specified table/column. (Requires safe_sql_identifier and text)"""
    safe_col = safe_sql_identifier(col)
    with engine.begin() as conn:
        conn.execute(
            text(f"CREATE INDEX {idx_name} ON {safe_sql_identifier(table)}({safe_col})")
        )


def drop_index(engine, table, idx_name, db_mode):
    """Drops the index based on the DB mode. (Requires safe_sql_identifier and text)"""
    with engine.begin() as conn:
        if db_mode == "MySQL":
            conn.execute(text(f"DROP INDEX {idx_name} ON {safe_sql_identifier(table)}"))
        else:
            conn.execute(text(f"DROP INDEX {idx_name}"))


# --- FIX for "render_table_preview not found" error ---
def render_table_preview(engine, selected_table, num_rows=10):
    """Fetches and displays a small preview of the selected table."""
    import streamlit as st
    import pandas as pd

    if not selected_table or selected_table == "(no tables found)":
        return

    try:
        # Assuming safe_sql_identifier is a globally available function (or you define it)
        query = f"SELECT * FROM {safe_sql_identifier(selected_table)} LIMIT {num_rows}"
        df_preview = pd.read_sql(query, engine)

        with st.expander(f"Data Preview: {selected_table} (First {num_rows} rows)", expanded=False):
            st.dataframe(df_preview, width='stretch')

    except Exception:
        # Fails silently if the table is not queryable (e.g., disconnected)
        pass

# --------------------------------------------------
# End of Global Helper Functions
# --------------------------------------------------

# Ensure safe_sql_identifier, text, execute_and_measure, log_performance_db, get_table_metadata are defined elsewhere.

import numpy as np  # Ensure this is imported globally
import pandas as pd  # Ensure this is imported globally
import streamlit as st  # Ensure this is imported globally
import matplotlib.pyplot as plt  # Ensure this is imported globally
import hashlib  # Ensure this is imported globally
import os  # Ensure this is imported globally
import re  # Ensure this is imported globally
from sqlalchemy import text  # Ensure this is imported globally


# NOTE: This function assumes the following helper functions are defined in the global scope:
# render_table_preview, safe_sql_identifier, execute_and_measure, log_performance_db,
# clean_index_column, get_table_metadata, create_index, drop_index.

def render_research_lab():
    """Renders the Research Lab (Performance, Indexing, Benchmarking) tab with improved UI."""

    # Define table name early to avoid shadowing errors
    selected_table_name = st.session_state.get("selected_table", "(no tables found)")

    if st.session_state.get("connected") and selected_table_name != "(no tables found)":
        engine = st.session_state["engine"]
        render_table_preview(engine, selected_table_name, num_rows=10)

    st.header("🔬 Database Research Lab — Performance & Indexing")

    st.info("""
        This lab tests how physical design (indexing, table size) affects performance (time, memory).
        ⚠️ **Warning:** Experiments can create and drop temporary tables. Use this on test/sample data only.
    """)

    if not st.session_state.get("connected") or selected_table_name == "(no tables found)":
        st.warning("Please connect to a database and select a table in the **Connection & Setup** tab first.")
        return

    engine = st.session_state["engine"]
    db_mode = st.session_state["db_mode_choice"]

    # --- 1. Query Performance & EXPLAIN ---
    with st.expander("1️⃣ Query Performance & EXPLAIN (Profiling)", expanded=True):
        st.markdown("Measure execution time and memory for custom queries, and view the query plan.")

        query_default = f"SELECT * FROM {safe_sql_identifier(selected_table_name)} LIMIT 1000"
        query_text = st.text_area("SELECT query to run", value=query_default, height=100, key="perf_query_text_area")

        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_a:
            sim_enable = st.checkbox("Simulate larger dataset (SQLite only)", value=False, key="sim_dup")
        with col_b:
            dup_factor = st.number_input("Duplication factor", min_value=2, max_value=200, value=5,
                                         key="dup_factor_mod1",
                                         disabled=not sim_enable or not db_mode.startswith("SQLite"))
        with col_c:
            st.markdown("<br>", unsafe_allow_html=True)
            run_query_btn = st.button("▶️ Run & Measure Query", key="run_measure_btn", type="primary")

        temp_table = f"tmp_sim_{selected_table_name}_{hashlib.sha1(os.urandom(10)).hexdigest()[:5]}"

        if run_query_btn:
            result_container = st.container()

            if sim_enable and db_mode.startswith("SQLite"):
                st.info(
                    f"Simulating large dataset: Creating temp table `{temp_table}` by duplicating data {dup_factor} times...")
                try:
                    with engine.begin() as conn:
                        conn.execute(text(f"DROP TABLE IF EXISTS {safe_sql_identifier(temp_table)}"))
                        conn.execute(text(
                            f"CREATE TABLE {safe_sql_identifier(temp_table)} AS SELECT * FROM {safe_sql_identifier(selected_table_name)} WHERE 1=0"))
                        insert_query = f"INSERT INTO {safe_sql_identifier(temp_table)} SELECT * FROM {safe_sql_identifier(selected_table_name)}"
                        for i in range(int(dup_factor)):
                            conn.execute(text(insert_query))

                    sim_query = query_text.replace(safe_sql_identifier(selected_table_name),
                                                   safe_sql_identifier(temp_table))
                    df_result, time_elapsed, memory_peak = execute_and_measure(sim_query, engine)
                    log_performance_db(db_mode, st.session_state["selected_db_name"], query_text, len(df_result),
                                       time_elapsed, memory_peak, indexed=0)
                    st.success(f"Execution Complete (Simulated on `{temp_table}`)")

                except Exception as e:
                    st.error(f"❌ Simulation or Query execution failed: {e}")
                    return

                finally:
                    try:
                        with engine.begin() as conn:
                            conn.execute(text(f"DROP TABLE IF EXISTS {safe_sql_identifier(temp_table)}"))
                    except:
                        st.warning("Failed to drop temporary table. You may need to drop it manually.")

                final_query = sim_query

            else:  # Standard execution
                try:
                    df_result, time_elapsed, memory_peak = execute_and_measure(query_text, engine)
                    log_performance_db(db_mode, st.session_state["selected_db_name"], query_text, len(df_result),
                                       time_elapsed, memory_peak, indexed=0)
                    st.success("Execution Complete")
                except Exception as e:
                    st.error(f"❌ Query execution failed: {e}")
                    return

                final_query = query_text

            # --- Display Metrics ---
            with result_container:
                col1, col2, col3 = st.columns(3)
                col1.metric("Execution Time", f"{time_elapsed:.4f}s")
                col2.metric("Peak Memory", f"{memory_peak:.2f} MB")
                col3.metric("Rows Returned", f"{len(df_result):,}")

                st.subheader("Results Table (First 20 Rows)")
                # FIX: use_container_width -> width='stretch'
                st.dataframe(df_result.head(20), width='stretch')

                # --- Explain Plan ---
                st.subheader("Query Plan (EXPLAIN)")
                try:
                    if db_mode == "MySQL":
                        explain_df = pd.read_sql(f"EXPLAIN {final_query}", engine)
                    else:  # SQLite
                        explain_df = pd.read_sql(f"EXPLAIN QUERY PLAN {final_query}", engine)
                    # FIX: use_container_width -> width='stretch'
                    st.dataframe(explain_df, width='stretch')
                except Exception as e:
                    st.error(f"Failed to fetch EXPLAIN plan: {e}")

    st.divider()

    # --- 2. Indexing Techniques & Tester ---
    with st.expander("2️⃣ Indexing Techniques & Tester (Index vs. Scan)", expanded=False):
        st.markdown("Compare query performance before and after creating a temporary index.")
        st.info("""
            **How to Test:** Enter the column name and a test query (which **must** filter on that column). The app runs the query in three phases: 1. No Index, 2. Index Created, 3. Index Dropped (Cleanup).
        """)

        # --- Input Fields ---
        col_index_input, col_test_query = st.columns(2)

        with col_index_input:
            initial_index_value = st.session_state.get("index_col_input", "")

            if "_ai_suggested_index_col" in st.session_state:
                initial_index_value = clean_index_column(st.session_state.pop("_ai_suggested_index_col"))
                st.session_state["index_col_input"] = initial_index_value

            index_column_raw = st.text_input(
                "Column to index (exact name)",
                value=initial_index_value,
                key="index_col_input"
            )
            index_column = clean_index_column(index_column_raw)

            safe_fallback_col = "id"
            if index_column:
                try:
                    col_metadata = get_table_metadata(engine, selected_table_name)
                    safe_fallback_col = next(iter(col_metadata.keys()), 'id')

                    if index_column not in col_metadata:
                        index_column = safe_fallback_col
                        if index_column != st.session_state.get("index_col_input"):
                            st.session_state["index_col_input"] = index_column
                            # FIX: Correct use of st.rerun()
                            st.rerun()
                except Exception:
                    pass

        with col_test_query:
            index_test_query = st.text_area(
                "Test query (MUST use the indexed column in WHERE)",
                value=f"SELECT * FROM {safe_sql_identifier(selected_table_name)} WHERE {safe_sql_identifier(index_column)} IS NOT NULL LIMIT 1000",
                height=80,
                key="index_test_query_area"
            )

        if st.button("🚀 Run Indexing Comparison", key="run_index_test_btn", type="primary"):
            if not index_column or not index_test_query.strip():
                st.error("Provide a valid column name and a test query.")
                return

            idx_name = f"idx_test_{index_column}_{hashlib.sha1(os.urandom(10)).hexdigest()[:5]}"
            results = []

            status_bar = st.empty()

            # 1. No Index Run
            status_bar.info("Phase 1/3: Running query **without** index...")
            try:
                df_no, t_no, m_no = execute_and_measure(index_test_query, engine)
                results.append({"Run": "No Index", "Time (s)": t_no})
                log_performance_db(db_mode, st.session_state["selected_db_name"], index_test_query, len(df_no), t_no,
                                   m_no, indexed=0)
            except Exception as e:
                status_bar.error(f"No Index Run Failed: {e}")
                return

            # 2. Create Index
            status_bar.info(f"Phase 2/3: Creating index `{idx_name}` on column `{index_column}`...")
            try:
                create_index(engine, selected_table_name, index_column, idx_name)
            except Exception as e:
                status_bar.error(f"Index Creation Failed: {e}. Aborting test.")
                try:
                    drop_index(engine, selected_table_name, idx_name, db_mode)
                except:
                    pass
                return

            # 3. Index Run
            status_bar.info(f"Phase 2/3: Running query **with** index `{idx_name}`...")
            try:
                df_idx, t_idx, m_idx = execute_and_measure(index_test_query, engine)
                results.append({"Run": "With Index", "Time (s)": t_idx})
                log_performance_db(db_mode, st.session_state["selected_db_name"], index_test_query, len(df_idx), t_idx,
                                   m_idx, indexed=1)
            except Exception as e:
                status_bar.error(f"Indexed Run Failed: {e}")
                t_idx = t_no

            # 4. Drop Index (Cleanup)
            status_bar.info("Phase 3/3: Dropping index (Cleanup)...")
            try:
                drop_index(engine, selected_table_name, idx_name, db_mode)
                status_bar.success("Indexing Comparison Complete. Index dropped. Environment restored.")
            except Exception as e:
                status_bar.error(f"Index Drop Failed: {e}. Please drop index `{idx_name}` manually.")

            # --- Results Visualization ---
            df_r = pd.DataFrame(results).set_index("Run")
            st.subheader("Indexing Comparison Results")

            col_metric, col_chart = st.columns([1, 2])

            col_metric.metric("Performance Change",
                              f"{df_r.loc['No Index', 'Time (s)'] / df_r.loc['With Index', 'Time (s)']:.2f}x",
                              delta=f"Improvement from {df_r.loc['No Index', 'Time (s)']:.4f}s to {df_r.loc['With Index', 'Time (s)']:.4f}s")

            with col_chart:
                fig, ax = plt.subplots(figsize=(6, 3))
                df_r["Time (s)"].plot(kind='bar', ax=ax, color=['#e74c3c', '#27ae60'])
                ax.set_title("Query Time: No Index vs. With Index")
                ax.set_ylabel("Execution Time (s)")
                ax.tick_params(axis='x', rotation=0)
                st.pyplot(fig)
                plt.close(fig)

            st.markdown(
                """
                > **Research Insight:** The performance difference demonstrates the efficiency gain 
                > of avoiding a full table scan and using an optimized index structure.
                """
            )

    st.divider()

    # --- 3. Systematic Benchmark ---
    with st.expander("3️⃣ Systematic Benchmark (Time vs. Data Size)", expanded=False):
        st.write("Test how execution time and memory consumption scale as the query result size increases.")

        bench_query_default = f"SELECT * FROM {safe_sql_identifier(selected_table_name)}"
        bench_query = st.text_area("Base Query (No LIMIT or ORDER BY)", value=bench_query_default, height=80,
                                   key="bench_query_area")

        col_sizes, col_btn_bench = st.columns([2, 1])
        with col_sizes:
            sizes_input = st.multiselect("Select result sample counts to benchmark",
                                         options=[100, 1000, 5000, 10000, 50000, 100000],
                                         default=[1000, 5000, 10000], key="bench_sizes")
        with col_btn_bench:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📊 Run Systematic Benchmark", key="bench_btn", type="primary"):
                if not bench_query.strip():
                    st.error("Provide a query.")
                    return

                results = []
                st.subheader("Benchmark Results")
                progress_container = st.container()

                with progress_container:
                    progress_bar = st.progress(0, text="Starting Benchmark...")

                for i, sz in enumerate(sizes_input):
                    sq = f"{bench_query} LIMIT {sz}"
                    progress_bar.progress((i + 1) / len(sizes_input), text=f"Running query for {sz:,} rows...")
                    try:
                        df_sz, t_sz, m_sz = execute_and_measure(sq, engine)
                        results.append({"rows": sz, "time": t_sz, "mem_mb": m_sz})
                        log_performance_db(db_mode, st.session_state["selected_db_name"], bench_query, len(df_sz), t_sz,
                                           m_sz, indexed=0)
                    except Exception as e:
                        st.warning(f"Failed at size {sz}: {e}")

                progress_bar.empty()
                st.success("Benchmark completed.")

                if results:
                    df_r = pd.DataFrame(results).set_index("rows")
                    col_time, col_mem = st.columns(2)
                    with col_time:
                        st.subheader("Time vs. Data Size Trend")
                        st.line_chart(df_r["time"])
                    with col_mem:
                        st.subheader("Memory vs. Data Size Trend")
                        st.line_chart(df_r["mem_mb"])

    st.divider()

    # --- 4. Query Optimizer Cost Model Visualizer ---
    with st.expander("4️⃣ Query Optimizer Cost Model Visualizer 🧠", expanded=False):
        st.write("Visualize the estimated cost versus selectivity, demonstrating when the execution plan changes.")
        st.info("""
            **Research Insight:** The crossover point on the chart shows when the Query Optimizer flips from an **Index Scan** (low selectivity) to a full **Table Scan** (high selectivity).
        """)

        col_metadata = get_table_metadata(engine, selected_table_name)
        col_names = list(col_metadata.keys())

        if not col_names:
            st.warning("No columns found in the selected table to run the optimizer simulation.")
            return

        default_col = next((c for c in col_names if 'price' in c.lower() or 'id' in c.lower()),
                           col_names[0] if col_names else None)
        optim_col = st.selectbox("Column for Selectivity (Numeric only for best results)", col_names,
                                 index=col_names.index(default_col) if default_col in col_names else 0, key="optim_col")

        if st.button("▶️ Run Optimizer Cost Simulation (10 Steps)", key="run_optimizer_sim", type="primary"):
            if not optim_col:
                st.error("Please select a column.")
                return

            try:
                with engine.connect() as conn:
                    min_val_row = conn.execute(text(
                        f"SELECT MIN({safe_sql_identifier(optim_col)}) FROM {safe_sql_identifier(selected_table_name)}")).fetchone()
                    max_val_row = conn.execute(text(
                        f"SELECT MAX({safe_sql_identifier(optim_col)}) FROM {safe_sql_identifier(selected_table_name)}")).fetchone()

                min_val_raw = min_val_row[0] if min_val_row and min_val_row[0] is not None else 0
                max_val_raw = max_val_row[0] if max_val_row and max_val_row[0] is not None else 100

                # CRITICAL FIX: Explicitly convert Decimal to float for NumPy compatibility
                min_val = float(min_val_raw)
                max_val = float(max_val_raw)

            except Exception as e:
                st.error(f"Could not retrieve MIN/MAX for column: {e}")
                return

            if max_val <= min_val:
                st.error("Column range is too small or contains non-numeric data. Please choose a numeric column.")
                return

            num_steps = 10
            # ERROR FIX: This now works because min_val and max_val are floats
            thresholds = np.linspace(min_val, max_val, num_steps)
            results = []

            progress = st.progress(0, text="Analyzing Query Plan Costs...")

            for i, threshold in enumerate(thresholds):
                sim_query = f"SELECT * FROM {safe_sql_identifier(selected_table_name)} WHERE {safe_sql_identifier(optim_col)} > {threshold}"

                cost_proxy = 0
                plan = 'N/A'

                try:
                    if db_mode == "MySQL":
                        explain_q = "EXPLAIN " + sim_query
                        exdf = pd.read_sql_query(explain_q, engine)
                        cost_proxy = exdf['rows'].sum() if not exdf.empty and 'rows' in exdf.columns else 0
                        plan = exdf['type'].iloc[0] if not exdf.empty and 'type' in exdf.columns else 'N/A'
                    else:  # SQLite
                        explain_q = "EXPLAIN QUERY PLAN " + sim_query
                        exdf = pd.read_sql_query(explain_q, engine)

                        for row in exdf['detail']:
                            plan = 'Index Scan' if 'INDEX' in row.upper() else 'Table Scan'
                            match_row_count = re.search(r'rows=(\d+)', row)
                            if match_row_count:
                                cost_proxy = max(cost_proxy, int(match_row_count.group(1)))

                    selectivity = (max_val - threshold) / (max_val - min_val) * 100 if (max_val - min_val) != 0 else 0
                    results.append({"Selectivity (%)": selectivity, "Cost Proxy (Rows)": cost_proxy, "Plan": plan})

                except Exception as e:
                    st.warning(f"Failed to explain query at threshold {threshold}: {e}")
                    results.append(
                        {"Selectivity (%)": (max_val - threshold) / (max_val - min_val) * 100, "Cost Proxy (Rows)": 0,
                         "Plan": "Error"})

                progress.progress((i + 1) / num_steps)

            progress.empty()
            st.success("Simulation complete.")

            df_r = pd.DataFrame(results).sort_values("Selectivity (%)", ascending=True)

            st.subheader("Optimizer Cost vs. Selectivity Chart")

            fig, ax = plt.subplots(figsize=(10, 5))
            for plan_name, group in df_r.groupby("Plan"):
                ax.plot(group["Selectivity (%)"], group["Cost Proxy (Rows)"], marker='o', label=plan_name)

            ax.set_xlabel(f"Query Selectivity (%) - (e.g., {optim_col} > X)")
            ax.set_ylabel("Estimated Cost Proxy (Rows Processed)")
            ax.set_title("Optimizer Decision Model Visualization")
            ax.legend(title="Execution Plan")
            ax.grid(True)
            st.pyplot(fig)
            plt.close(fig)

            st.markdown(
                """
                > **Research Insight:** The performance difference demonstrates the efficiency gain 
                > of avoiding a full table scan and using an optimized index structure.
                """
            )

    st.divider()

    # --- 4. Query Optimizer Cost Model Visualizer (Unchanged) ---
    with st.expander("4️⃣ Query Optimizer Cost Model Visualizer 🧠", expanded=False):
        st.write("Visualize the estimated cost versus selectivity, demonstrating when the execution plan changes.")
        st.info("""
            **Research Insight:** The crossover point on the chart shows when the Query Optimizer flips from an **Index Scan** (low selectivity) to a full **Table Scan** (high selectivity).
        """)

        col_metadata = get_table_metadata(engine, selected_table)
        col_names = list(col_metadata.keys())

        if not col_names:
            st.warning("No columns found in the selected table to run the optimizer simulation.")
            return

        default_col = next((c for c in col_names if 'price' in c.lower() or 'id' in c.lower()),
                           col_names[0] if col_names else None)
        optim_col = st.selectbox("Column for Selectivity (Numeric only for best results)", col_names,
                                 index=col_names.index(default_col) if default_col in col_names else 0, key="optim_col")

        if st.button("▶️ Run Optimizer Cost Simulation (10 Steps)", key="run_optimizer_sim", type="primary"):
            if not optim_col:
                st.error("Please select a column.")
                return

            try:
                with engine.connect() as conn:
                    min_val_row = conn.execute(text(
                        f"SELECT MIN({safe_sql_identifier(optim_col)}) FROM {safe_sql_identifier(selected_table)}")).fetchone()
                    max_val_row = conn.execute(text(
                        f"SELECT MAX({safe_sql_identifier(optim_col)}) FROM {safe_sql_identifier(selected_table)}")).fetchone()

                min_val = min_val_row[0] if min_val_row and min_val_row[0] is not None else 0
                max_val = max_val_row[0] if max_val_row and max_val_row[0] is not None else 100

            except Exception as e:
                st.error(f"Could not retrieve MIN/MAX for column: {e}")
                return

            if max_val <= min_val:
                st.error("Column range is too small or contains non-numeric data. Please choose a numeric column.")
                return

            num_steps = 10
            thresholds = np.linspace(min_val, max_val, num_steps)
            results = []

            progress = st.progress(0, text="Analyzing Query Plan Costs...")

            for i, threshold in enumerate(thresholds):
                sim_query = f"SELECT * FROM {safe_sql_identifier(selected_table)} WHERE {safe_sql_identifier(optim_col)} > {threshold}"

                cost_proxy = 0
                plan = 'N/A'

                try:
                    if db_mode == "MySQL":
                        explain_q = "EXPLAIN " + sim_query
                        exdf = pd.read_sql_query(explain_q, engine)
                        cost_proxy = exdf['rows'].sum() if not exdf.empty and 'rows' in exdf.columns else 0
                        plan = exdf['type'].iloc[0] if not exdf.empty and 'type' in exdf.columns else 'N/A'
                    else:  # SQLite
                        explain_q = "EXPLAIN QUERY PLAN " + sim_query
                        exdf = pd.read_sql_query(explain_q, engine)

                        for row in exdf['detail']:
                            plan = 'Index Scan' if 'INDEX' in row.upper() else 'Table Scan'
                            match = re.search(r'rows=(\d+)', row)
                            if match:
                                cost_proxy = max(cost_proxy, int(match.group(1)))

                    selectivity = (max_val - threshold) / (max_val - min_val) * 100 if (max_val - min_val) != 0 else 0
                    results.append({"Selectivity (%)": selectivity, "Cost Proxy (Rows)": cost_proxy, "Plan": plan})

                except Exception as e:
                    st.warning(f"Failed to explain query at threshold {threshold}: {e}")
                    results.append(
                        {"Selectivity (%)": (max_val - threshold) / (max_val - min_val) * 100, "Cost Proxy (Rows)": 0,
                         "Plan": "Error"})

                progress.progress((i + 1) / num_steps)

            progress.empty()
            st.success("Simulation complete.")

            df_r = pd.DataFrame(results).sort_values("Selectivity (%)", ascending=True)

            st.subheader("Optimizer Cost vs. Selectivity Chart")

            fig, ax = plt.subplots(figsize=(10, 5))
            for plan, group in df_r.groupby("Plan"):
                ax.plot(group["Selectivity (%)"], group["Cost Proxy (Rows)"], marker='o', label=plan)

            ax.set_xlabel(f"Query Selectivity (%) - (e.g., {optim_col} > X)")
            ax.set_ylabel("Estimated Cost Proxy (Rows Processed)")
            ax.set_title("Optimizer Decision Model Visualization")
            ax.legend(title="Execution Plan")
            ax.grid(True)
            st.pyplot(fig)
            plt.close(fig)

        st.markdown(
            """
            > **Research Insight:** Notice the crossover point. When the selectivity is **low** (few rows returned), the plan is usually an **Index Scan** (low cost). 
            > When selectivity is **high** (many rows returned), the cost of traversing the index is higher than a full **Table Scan**, and the optimizer switches plans.
            """
        )

def render_nosql_lab():
    """Renders the NoSQL and Distributed Systems research tab."""
    st.header("🌐 NoSQL & Distributed Systems Lab")

    # -------------------------
    # TUTORIAL TEXT START
    # -------------------------
    st.markdown("""
        This lab explores key architectural concepts used in modern distributed and NoSQL databases, focusing on performance trade-offs under different workloads.

        * **OLAP vs. OLTP:** Demonstrates the difference between fast single-row lookups and complex aggregations.
        * **Storage Engines:** Compares the I/O costs of B-Trees (standard SQL index) versus LSM-Trees (common in NoSQL like Cassandra, MongoDB).
        * **Sharding:** Visualizes how data is distributed across multiple servers to handle high load.
    """)
    # -------------------------
    # TUTORIAL TEXT END
    # -------------------------

    # Check connection just in case a table-dependent feature is run
    if not st.session_state["connected"] or st.session_state["selected_table"] == "(no tables found)":
        st.warning(
            "Please connect to a database and select a table in the **Connection & Setup** tab first for the table-based simulations.")
        # Continue rendering non-dependent sections...

    engine = st.session_state["engine"]
    selected_table = st.session_state["selected_table"]

    # -------------------------
    # 1. OLAP vs. OLTP Comparison
    # -------------------------
    st.markdown("## 1. OLAP vs. OLTP Comparison (Data Architecture)")
    st.write(
        "Demonstrate the difference between fast **Transactional (OLTP)** lookups and complex **Analytical (OLAP)** aggregations, "
        "highlighting the architectural trade-offs (row vs. column storage)."
    )

    if st.button("▶️ Run OLAP/OLTP Comparison Test (Requires Sample Data)", key="run_olap_oltp"):
        if not st.session_state["connected"] or st.session_state["selected_table"] == "(no tables found)":
            st.warning("Please connect to a database and select a table first.")
            return

        # Simple heuristic to pick columns for the queries
        col_metadata = get_table_metadata(engine, selected_table)
        pk_col = next(iter(col_metadata.keys()), 'id')  # Assume first column is PK-like

            # 1. Determine Aggregate Column (agg_col)
            # Prioritize 'amount' or 'price', then fallback to a default numeric column.
        agg_col = next((c for c in col_metadata.keys() if 'price' in c.lower() or 'amount' in c.lower()), None)
        if agg_col is None:
                # Fallback to the first numeric column found (e.g., 'qty')
            agg_col = next((c for c, dtype in col_metadata.items() if 'int' in dtype.lower() or 'float' in dtype.lower()),
                    'id')

            # 2. Determine Grouping Column (group_col)
            # Prioritize 'category' or 'region', then fallback to 'item_id', then to the primary key.
        group_col = next((c for c in col_metadata.keys() if 'category' in c.lower() or 'region' in c.lower()), None)
        if group_col is None:
                # Fallback for tables like 'sales' which have 'item_id'
            group_col = next((c for c in col_metadata.keys() if c.lower() == 'item_id'), pk_col)

                # Safety check: ensure the group_col is not the same as the aggregate column if possible
        if group_col == agg_col and len(col_metadata) > 1:
            group_col = next((c for c in col_metadata.keys() if c != agg_col), pk_col)

        # Generate a random ID for OLTP lookup (need to ensure it exists)
        try:
            with engine.connect() as conn:
                result = conn.execute(text(
                    f"SELECT {safe_sql_identifier(pk_col)} FROM {safe_sql_identifier(selected_table)} LIMIT 1 OFFSET {random.randint(0, 9)}")).fetchone()
                oltp_id = result[0] if result else 1
        except Exception:
            oltp_id = 1
            st.warning("Could not sample data; using default ID=1 for OLTP lookup.")

        # 1. OLTP Query (Lookup)
        oltp_query = f"SELECT * FROM {safe_sql_identifier(selected_table)} WHERE {safe_sql_identifier(pk_col)} = {oltp_id}"

        # 2. OLAP Query (Aggregation)
        olap_query = f"SELECT {safe_sql_identifier(group_col)}, SUM({safe_sql_identifier(agg_col)}) AS total_sales FROM {safe_sql_identifier(selected_table)} GROUP BY {safe_sql_identifier(group_col)}"

        st.subheader("Query Definitions")
        col_o1, col_o2 = st.columns(2)
        with col_o1:
            st.caption("OLTP (Transactional) Query")
            st.code(oltp_query, language="sql")
        with col_o2:
            st.caption("OLAP (Analytical) Query")
            st.code(olap_query, language="sql")

        # Execute and Measure
        st.subheader("Execution Results")
        try:
            df_oltp, t_oltp, m_oltp = execute_and_measure(oltp_query, engine)
            st.success(f"OLTP Query Time: **{t_oltp:.6f}s** (Fast single-row lookup)")
        except Exception as e:
            st.error(f"OLTP test failed: {e}")
            t_oltp = 0.5  # Default fail value

        try:
            df_olap, t_olap, m_olap = execute_and_measure(olap_query, engine)
            st.success(f"OLAP Query Time: **{t_olap:.4f}s** (Slower full-table aggregation)")
        except Exception as e:
            st.error(f"OLAP test failed: {e}")
            t_olap = 5.0  # Default fail value

        # Visualization
        fig, ax = plt.subplots(figsize=(6, 4))
        times = [t_oltp * 1000, t_olap]  # Show OLTP in milliseconds for contrast
        labels = ['OLTP (ms)', 'OLAP (s)']
        ax.bar(labels, times, color=['#27ae60', '#3498db'])
        ax.set_ylabel("Execution Time (s)")
        ax.set_yscale('log')  # This is the correct way to set a logarithmic scale
        ax.set_title("OLTP vs. OLAP Performance")
        st.pyplot(fig)

        st.markdown(
            f"""
            > **Architectural Insights:**
            > * **OLTP:** (e.g., `{pk_col}` lookup) is usually extremely fast, focused on **single rows** and write consistency. Optimized by **Row Storage** and Primary Indexes.
            > * **OLAP:** (e.g., `SUM({agg_col})` group by `{group_col}`) is typically much slower, focused on **many columns/rows** for complex reads. Optimized by **Column Storage** and data warehousing techniques.
            """
        )

    # -------------------------
    # 2. Storage Engine Trade-offs (B-Tree vs. LSM-Tree)
    # -------------------------
    st.markdown("---")
    st.markdown("## 2. Storage Engine Trade-offs 🗃️ (B-Tree vs. LSM-Tree)")
    st.info("""
        **B-Tree:** Excellent for reads, good for updates, but writes can be expensive (in-place page modification). Common in relational DBs (InnoDB).
        **LSM-Tree:** Excellent for high-throughput writes (append-only), but reads can be expensive (must check multiple sorted layers). Common in NoSQL (Cassandra, MongoDB).
    """)
    st.write(
        "Simulate the conceptual I/O performance of **B-Trees** (Relational) and **LSM-Trees** (NoSQL) "
        "under different workloads (Reads vs. Writes)."
    )

    num_ops = st.slider("Number of Random Operations (Simulated)", 100, 2000, 500, step=100, key="io_scale")

    if st.button("▶️ Run Storage Engine I/O Simulation", key="run_io_sim"):
        # Conceptual I/O Cost Factors (Simulated)
        # B-Tree: High cost for random write (page overwrite), low cost for random read (logarithmic lookup)
        # LSM-Tree: Low cost for random write (sequential append), high cost for random read (checking multiple segments)
        BTREE_READ_COST = 5
        BTREE_WRITE_COST = 25
        LSM_READ_COST = 15
        LSM_WRITE_COST = 2

        # --- High Read Mode Simulation (80% Read / 20% Write) ---
        reads_hr = int(num_ops * 0.8)
        writes_hr = int(num_ops * 0.2)

        # Calculate costs for High Read
        bt_read_hr = reads_hr * BTREE_READ_COST
        bt_write_hr = writes_hr * BTREE_WRITE_COST
        lsm_read_hr = reads_hr * LSM_READ_COST
        lsm_write_hr = writes_hr * LSM_WRITE_COST

        df_hr = pd.DataFrame({
            'Engine': ['B-Tree (SQL)', 'LSM-Tree (NoSQL)'],
            'Read I/O': [bt_read_hr, lsm_read_hr],
            'Write I/O': [bt_write_hr, lsm_write_hr],
            'Total I/O': [bt_read_hr + bt_write_hr, lsm_read_hr + lsm_write_hr]
        }).set_index('Engine')

        # --- High Write Mode Simulation (20% Read / 80% Write) ---
        reads_hw = int(num_ops * 0.2)
        writes_hw = int(num_ops * 0.8)

        # Calculate costs for High Write
        bt_read_hw = reads_hw * BTREE_READ_COST
        bt_write_hw = writes_hw * BTREE_WRITE_COST
        lsm_read_hw = reads_hw * LSM_READ_COST
        lsm_write_hw = writes_hw * LSM_WRITE_COST

        df_hw = pd.DataFrame({
            'Engine': ['B-Tree (SQL)', 'LSM-Tree (NoSQL)'],
            'Read I/O': [bt_read_hw, lsm_read_hw],
            'Write I/O': [bt_write_hw, lsm_write_hw],
            'Total I/O': [bt_read_hw + bt_write_hw, lsm_read_hw + lsm_write_hw]
        }).set_index('Engine')

        # Visualization
        fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        x = np.arange(len(df_hr.index))
        width = 0.35
        x_labels = df_hr.index

        # Plot 1: High Read
        rects1 = ax[0].bar(x - width / 2, df_hr['Read I/O'], width, label='Read I/O', color='#3498db')
        rects2 = ax[0].bar(x + width / 2, df_hr['Write I/O'], width, label='Write I/O', color='#e74c3c')
        ax[0].set_ylabel('Simulated Disk Operations')
        ax[0].set_title('High Read Workload (80% R)')
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(x_labels)
        ax[0].legend()

        # Plot 2: High Write
        rects3 = ax[1].bar(x - width / 2, df_hw['Read I/O'], width, label='Read I/O', color='#3498db')
        rects4 = ax[1].bar(x + width / 2, df_hw['Write I/O'], width, label='Write I/O', color='#e74c3c')
        ax[1].set_title('High Write Workload (80% W)')
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(x_labels)
        ax[1].legend()

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown(
            """
            > **Research Insight:**
            > * In **High Read** scenarios, the B-Tree has lower I/O cost because it can locate data quickly with minimal disk seeks.
            > * In **High Write** scenarios, the LSM-Tree is far superior because all writes are sequential appends to memory/log files (lower I/O cost), minimizing disk seeks. The trade-off is that LSM-Tree's reads often require checking multiple layers, increasing its read cost.
            """
        )

    # -------------------------
    # 3. Sharding / Partitioning Visualizer
    # -------------------------
    st.markdown("---")
    st.markdown("## 3. Sharding/Partitioning Visualizer (Distributed Systems)")
    st.info("""
        **What it shows:** Sharding distributes data across multiple independent servers (shards) to handle more load than a single machine. The **Sharding Key** determines which server a row goes to. A poor key can lead to **Hotspots** (data piling up on one server).
    """)
    st.write(
        "Visualize how data distribution strategies (sharding) work across a distributed cluster "
        "using a chosen **sharding key**."
    )

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        shards = st.slider("Number of Logical Shards", 2, 8, 4, key="num_shards")
    with col_s2:
        sharding_key = st.text_input("Simulated Sharding Key Column (e.g., user_id or category)", value="category",
                                     key="sharding_key_input")

    if st.button("▶️ Run Sharding Simulation", key="run_sharding_sim"):
        if not st.session_state["connected"] or st.session_state["selected_table"] == "(no tables found)":
            st.warning("Please connect to a database and select a table first.")
            return

        try:
            # 1. Fetch data for the sharding key
            with engine.connect() as conn:
                df_data = pd.read_sql(
                    f"SELECT {safe_sql_identifier(sharding_key)} FROM {safe_sql_identifier(selected_table)}", conn)

            if df_data.empty:
                st.error("The selected table is empty.")
                return

            # 2. Simulate sharding (simple modulus on a hash for uniform distribution)
            # The goal is to show count per shard

            # --- Handle different data types for hashing ---
            if pd.api.types.is_numeric_dtype(df_data[sharding_key]):
                # Use the value itself if numeric
                key_values = df_data[sharding_key].fillna(0).astype(int)
            else:
                # Hash the string value for non-numeric keys
                key_values = df_data[sharding_key].fillna('').astype(str).apply(
                    lambda x: int(hashlib.sha1(x.encode()).hexdigest(), 16))

            # --- Shard assignment and aggregation ---
            df_data['shard_id'] = key_values.apply(lambda x: x % shards)
            shard_counts = df_data['shard_id'].value_counts().sort_index()

            # 3. Visualization
            st.subheader(f"Distribution of Records Across {shards} Shards (Key: `{sharding_key}`)")

            fig, ax = plt.subplots(figsize=(8, 5))
            shard_counts.plot(kind='bar', ax=ax, color='#f1c40f')
            ax.set_title("Data Distribution by Shard ID")
            ax.set_xlabel("Shard ID")
            ax.set_ylabel("Number of Records")
            ax.tick_params(axis='x', rotation=0)
            st.pyplot(fig)

            total_records = len(df_data)
            avg_records = total_records / shards

            st.info(f"Total Records: {total_records} | Average per Shard: {avg_records:.0f}")

            # Hotspot analysis
            max_shard = shard_counts.max()
            if max_shard > avg_records * 1.5:  # Simple heuristic for a "hotspot"
                st.error(
                    f"⚠️ **Hotspot Detected!** Shard {shard_counts.idxmax()} holds {max_shard} records, which is **{max_shard / avg_records:.1f}x** the average. This indicates an unbalanced load, which a poorly chosen sharding key (like a low-cardinality `category` column) can cause.")
            else:
                st.success(
                    "✅ Distribution appears relatively balanced. A good sharding key ensures even load distribution.")

        except Exception as e:
            st.error(f"Sharding simulation error: Ensure column '{sharding_key}' exists and is spelled correctly.")
            st.exception(e)


# --------------------------------------------------
# Main Application Flow (Tabbed Interface)
# --------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⚙️ Connection & Setup",
    "🚀 Dynamic SQL Builder",
    "🔬 Research Lab",
    "🌐 NoSQL & Distributed Systems",
    "🧠 AI Analyzer & Logs"
])

with tab1:
    render_connection_and_setup()

with tab2:
    render_dynamic_sql_builder()

with tab3:
    render_research_lab()

with tab4:
    render_nosql_lab()

with tab5:
    render_ai_analyzer_and_logs()

# -------------------------
# Footer / notes
# -------------------------
st.markdown("---")
st.markdown(
    "### Notes & Disclaimer\n"
    "- This tool is designed for **Database Research and Education** (e.g., your course, 'New Trends in Database Research').\n"
    "- Features like `Query Simulation` and `Indexing Experiments` are **destructive** (they create/drop test tables). Use them only on non-critical, test data.\n"
    "- Performance metrics (time, memory) are for **relative comparison** within this application environment, not absolute production benchmarks."
)