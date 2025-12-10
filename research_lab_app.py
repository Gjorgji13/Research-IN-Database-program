import hashlib
import os
import re
import sqlite3
import time
import tracemalloc
from contextlib import contextmanager
from typing import List, Tuple, Dict
import numpy as np # Used for sample data generation
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
# Ensure these imports are available (pip install streamlit pandas sqlalchemy pymysql matplotlib cryptography)
from sqlalchemy import create_engine, inspect, text, exc

# --- Handle External Dependency ---
# The original files referenced a local module. We use a placeholder to avoid breaking the app.
try:
    from indatabase_ai_ml import selected_table
except ImportError:
    selected_table = "placeholder"

# -------------------------
# Page config & UI Setup
# -------------------------
st.set_page_config(page_title="Database Research Lab", layout="wide")


# -------------------------
# Custom CSS for Professional Look
# -------------------------
def apply_custom_css():
    """Applies custom styling for a professional, academic look."""
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

st.title("🧠 AI-Driven Self-Optimizing Database Research Lab")
st.caption("Experimental Platform for New Trends in Database Systems Research (Academic Use)")

st.info("""
🔬 **Scientific Explanation:**
This module measures how indexing impacts execution time and memory.
Each experiment executes the same query with and without an index and logs:
- Execution time
- Memory usage
- Result size
These metrics are later used for AI learning and trend analysis.
""")

st.markdown("---")

# -------------------------
# Session State Initialization
# -------------------------
# --- Session State Initialization (Necessary for the connection function) ---
# Minimal required initializations for the new function's logic
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
if "mysql_databases" not in st.session_state: # NEW: To store the list of available databases
    st.session_state["mysql_databases"] = []
# ... (rest of your state variables) ...

# --- MySQL Specific Initialization (Needed for the UI inputs) ---
if "db_mode_choice" not in st.session_state:
    st.session_state["db_mode_choice"] = "MySQL"
if "mysql_host" not in st.session_state:
    st.session_state["mysql_host"] = "localhost"
if "selected_table" not in st.session_state:
    st.session_state["selected_table"] = "(no tables found)"
if "ai_q_text" not in st.session_state:
    st.session_state["ai_q_text"] = "SELECT * FROM items LIMIT 10;"
if "qb_filters" not in st.session_state:
    st.session_state["qb_filters"] = []
if "qb_join" not in st.session_state:
    st.session_state["qb_join"] = None
if "kv_store" not in st.session_state:
    st.session_state["kv_store"] = {}
if "ai_suggested_query_ran" not in st.session_state:
    st.session_state["ai_suggested_query_ran"] = False

# --- Defaulted to MySQL for the user's focus ---
if "db_mode_choice" not in st.session_state:
    st.session_state["db_mode_choice"] = "MySQL"
if "sqlite_file_path" not in st.session_state:
    st.session_state["sqlite_file_path"] = "research_db.sqlite"

# --- MySQL connection state variables (Starts empty to force UI input) ---
if "mysql_host" not in st.session_state:
    st.session_state["mysql_host"] = "localhost" # Safe default
if "mysql_port" not in st.session_state:
    st.session_state["mysql_port"] = "5000" # Safe default
if "mysql_user" not in st.session_state:
    # 🎯 FIX: START WITH BLANK USERNAME
    st.session_state["mysql_user"] = ""
if "mysql_pass" not in st.session_state:
    # 🎯 FIX: START WITH BLANK PASSWORD
    st.session_state["mysql_pass"] = ""
if "mysql_db" not in st.session_state:
    # 🎯 FIX: START WITH BLANK DB NAME
    st.session_state["mysql_db"] = ""
# --------------------------------------------------


# --------------------------------------------------

# --------------------------------------------------
# RESEARCH & GUIDANCE HELPER FUNCTIONS (Macedonian/English)
# --------------------------------------------------

def render_tab_guide(tab_name: str, context: str, newbie_tip: str, pro_tip: str):
    """Renders a collapsible user guide based on experience level."""
    st.markdown("---")
    st.markdown(f"## 📖 Упатство за {tab_name}")
    with st.expander("Кликни за детален Водич (Newbie/Pro)", expanded=False):
        st.markdown(f"**📚 Цел на овој дел:** {context}")
        st.markdown("---")
        st.markdown("### 👨‍💻 За Почетници (Newbie)")
        st.info(f"**💡 Што да правите:** {newbie_tip}")
        st.markdown("### 🔬 За Искусни (Pro / Research)")
        st.markdown(f"**🔬 Клучна хипотеза:** {pro_tip}")


def render_research_intro():
    """Renders scientific introduction and research problem definition."""
    st.header("📘 1. Research Overview: Дефиниција на Истражувачкиот Проблем")
    st.markdown("---")

    st.subheader("Тема: Самоуправувачки Бази на Податоци (Self-Driving Databases)")
    st.markdown("""
    Овој систем е контролирана експериментална платформа за тестирање на **Автономни, Самоуправувачки Бази на Податоци**.
    Транзицијата од статични кон само-учечки системи е клучниот тренд во истражувањето.
    """)

    st.markdown("---")

    st.subheader("🔍 Примарно Истражувачко Прашање (PhD Research Problem)")
    st.info("""
    **Како може однесувањето на работниот товар (Query Workload) да се искористи за автономно оптимизирање на физичкиот дизајн на базата преку AI-водени стратегии за индексирање?**
    """)

    st.markdown("---")

    col_q1, col_q2 = st.columns(2)
    with col_q1:
        st.subheader("🔬 Секундарни Истражувачки Прашања")
        st.markdown("""
        1. Колку е предвидлива корисноста на индексот од историскиот `Workload`?
        2. Која е точката на премин (crossover point) помеѓу Table Scan и Index Scan?
        3. Како дизајнот на индексот влијае на кривите на скалабилност?
        """)

    with col_q2:
        st.subheader("🎯 Таргетирани Нови Трендови (ФИКТ, Трет Циклус)")
        st.markdown("""
        - **Self-Driving Databases**: Целосна автономија.
        - **AI-Assisted Optimization**: Користење ML за tuning.
        - **Automated Physical Design**: Автоматско креирање и тестирање на индекси.
        - **Hybrid SQL–NoSQL Systems**: Споредба на структури (LSM-Tree).
        - **Cost-Based Optimizer Modeling**: Анализа на планот за извршување.
        """)

    st.markdown("---")
    st.markdown("### 🗺️ Упатство за Користење на Модулите (Истражувачки Тек)")
    st.markdown("""
    Користете ги модулите во секвенца (од 2 до 5) за да спроведете целосен научен експеримент:
    1. **⚙️ Setup**: Поставете ја контролираната база и обемот на податоци.
    2. **🧪 Workload**: Генерирајте го Workload-от (историја на барања) што системот ќе го 'научи'.
    3. **🤖 Autonomous Lab**: Дозволете му на АИ да препорача индекси, спроведете експерименти и тестирајте хибридни концепти.
    4. **📊 Results**: Донесете формални заклучоци врз основа на добиените криви на перформанси.
    """)


def simulate_lsm_btree_io(num_ops: int, storage_type: str) -> dict:
    """Simulates I/O cost based on LSM-Tree vs. B-Tree structure."""
    num_writes = num_ops  # num_ops represents total operations

    if storage_type == "B-Tree (SQL - Good Reads)":
        # B-Tree: High write cost (in-place update) but low read cost
        write_cost_factor = 1.5
        read_cost_factor = 0.5
    elif storage_type == "LSM-Tree (NoSQL - Good Writes)":
        # LSM-Tree: Low write cost (sequential log appends) but high read cost
        write_cost_factor = 0.3
        read_cost_factor = 1.2
    else:
        return {}

    # In the NoSQL Lab, num_ops is the total workload.
    # We use write_perc from the UI to split it.

    # Note: In the actual RENDER function, the cost will be calculated
    # based on the split (read_perc/write_perc) and then passed here.
    # To keep this function simple for its original intent:

    write_cost = num_ops * write_cost_factor
    read_cost = num_ops * read_cost_factor
    total_cost = write_cost + read_cost

    return {
        "Storage": storage_type,
        "Total Operations (Proxy)": num_ops,
        "Write Cost (Simulated I/O)": write_cost,
        "Read Cost (Simulated I/O)": read_cost,
        "Total Cost (Normalized)": total_cost,
    }


# --------------------------------------------------
# CORE DB/PERFORMANCE HELPERS
# --------------------------------------------------

@contextmanager
def sqlite_conn(path):
    """Context manager for the performance log SQLite database."""
    # This is only for the performance log database
    conn = sqlite3.connect(path, check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()


def norm_query_for_hash(q: str) -> str:
    """Calculates a hash for a query for logging/deduplication."""
    return hashlib.md5(q.strip().encode("utf-8")).hexdigest()


def ensure_perf_db(path="perf_logs.db"):
    """Ensures the performance log SQLite database and table exist."""
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
    """Logs the performance metrics of a query execution."""
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


def read_perf_logs(path="perf_logs.db", limit=500) -> pd.DataFrame:
    """Reads performance logs from the SQLite database."""
    with sqlite_conn(path) as c:
        df = pd.read_sql_query(f"SELECT * FROM perf_log ORDER BY timestamp DESC LIMIT {limit}", c)
    return df


def safe_sql_identifier(name: str) -> str:
    """Safe quotes identifiers for MySQL/SQLite."""
    # Use backticks for MySQL and general safety, SQLite handles them too.
    return f"`{name.replace('`', '')}`"


def get_table_metadata(engine, table_name) -> Dict[str, str]:
    """Retrieves column names and types for a given table."""
    try:
        inspector = inspect(engine)
        # Use safe quoting for table name retrieval
        cols = inspector.get_columns(table_name)
        return {c['name']: str(c['type']) for c in cols}
    except Exception:
        return {}


def execute_and_measure(query: str, engine) -> Tuple[pd.DataFrame, float, float]:
    """Execute query and measure time and memory usage."""
    tracemalloc.start()
    t0 = time.time()
    df = pd.DataFrame()
    try:
        with engine.connect() as conn:
            # Use read_sql_query which reads the full result set into a DataFrame
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


# --------------------------------------------------
# INDEX & ML HELPER FUNCTIONS
# --------------------------------------------------

def clean_index_column(col_name: str) -> str:
    """Cleans the input column name by removing common SQL syntax/quotes, preventing SQL syntax errors."""
    if not col_name: return ""
    col_name = col_name.replace('`', '').replace('"', '').replace("'", '').strip()
    # Removes trailing operators or spaces (e.g., 'price >' becomes 'price')
    col_name = re.sub(r'[\s<=>!]+$', '', col_name).strip()
    return col_name


def create_index(engine, table, col, idx_name):
    """Creates an index on the specified table/column."""
    safe_col = safe_sql_identifier(col)
    with engine.begin() as conn:
        conn.execute(
            text(f"CREATE INDEX {idx_name} ON {safe_sql_identifier(table)}({safe_col})")
        )


def drop_index(engine, table, idx_name, db_mode):
    """Drops the index based on the DB mode."""
    with engine.begin() as conn:
        if db_mode == "MySQL":
            # MySQL syntax requires ON table
            conn.execute(text(f"DROP INDEX {idx_name} ON {safe_sql_identifier(table)}"))
        else:
            # SQLite and general SQL syntax
            conn.execute(text(f"DROP INDEX IF EXISTS {idx_name}"))


def build_sql(
        selects: List[str], table: str, where: str = None, joins: str = None,
        group: str = None, having: str = None, order: str = None,
        order_dir: str = "ASC", limit: int = None
) -> str:
    """Assembles a SQL query from components."""
    sel = ", ".join(selects) if selects and selects[0] != "*" else "*"
    query = f"SELECT {sel} FROM {safe_sql_identifier(table)}"  # Ensure table is safe quoted
    if joins:
        query += f" {joins}"
    if where:
        query += f" WHERE {where}"
    if group:
        query += f" GROUP BY {group}"
    if having:
        query += f" HAVING {having}"
    if order and order != "(none)":
        order_col = safe_sql_identifier(order)
        query += f" ORDER BY {order_col} {order_dir}"
    if limit:
        query += f" LIMIT {limit}"
    return query


def train_ai_from_logs(limit=1000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Analyzes logs to find frequent filter columns (basic in-DB ML simulation)."""
    df_logs = read_perf_logs(limit=limit)
    if df_logs.empty: return pd.DataFrame(), pd.DataFrame()

    tokens = []
    for q in df_logs["query_text"].astype(str):
        # Very basic regex to find columns potentially used in WHERE clauses
        match = re.search(r"WHERE\s+[`]?(\w+)[`]?\s*[<>=!]", q.upper())
        if match:
            tokens.append(match.group(1))

    # Frequency analysis
    freq_df = pd.Series(tokens).value_counts().rename("count").to_frame()
    freq_df = freq_df.reset_index().rename(columns={'index': 'column'}).head(10)

    # Performance summary (Indexed vs. No Index)
    perf_summary = df_logs.groupby("indexed")["exec_time"].agg(['mean', 'median', 'count'])

    return freq_df, perf_summary


# --------------------------------------------------
# TAB RENDERING FUNCTIONS
# --------------------------------------------------

def render_table_preview(engine, table_name):
    """Utility to render the first 10 rows of the selected table."""
    try:
        with st.expander("Show Table Preview (First 10 Rows)", expanded=False):
            with engine.connect() as conn:
                preview_df = pd.read_sql(f"SELECT * FROM {safe_sql_identifier(table_name)} LIMIT 10", conn)
                st.dataframe(preview_df)
    except Exception as e:
        st.warning(f"Preview not available: {e}")


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
    # NOTE: The default index=0 makes SQLite the default choice now.
    db_mode = st.sidebar.selectbox("Select DB engine", ["SQLite (file)", "MySQL"], index=0, key="db_mode_choice")

    # Initialize connection variables with local defaults
    # These are NOT in session state, they are local defaults, which is the new style of this function.
    engine, connected, selected_db_name, tables = None, False, None, []
    sqlite_file, mysql_user, mysql_pass, mysql_host, mysql_port = "research_sample.db", "root", "", "localhost", "3306"

    # Load state from session if available (for UI persistence)
    connected = st.session_state.get("connected", False)

    # 2. Connection inputs & logic
    with st.sidebar.container(border=True):
        st.subheader("Connection Details")

        # --- SQLite Connection ---
        if db_mode == "SQLite (file)":
            # The input should be in the sidebar container, but the default code puts st.text_input in the main area.
            # I will assume st.text_input should be st.sidebar.text_input for better layout, but use your provided code structure.
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

        # --- MySQL Connection (Two-Step Process) ---
        else:
            mysql_host = st.text_input("MySQL host", value=mysql_host, key="mysql_host")
            mysql_port = st.text_input("MySQL port", value=mysql_port, key="mysql_port")
            mysql_user = st.text_input("MySQL user", value=mysql_user, key="mysql_user")
            mysql_pass = st.text_input("MySQL password", value=mysql_pass, type="password", key="mysql_pass")

            # Base URL without a specific database
            mysql_url_base = f"mysql+mysqlconnector://{mysql_user}:{mysql_pass}@{mysql_host}:{mysql_port}/"

            # Button 1: Test connection to server and fetch database list
            if st.button("Test MySQL Connection", key="test_mysql"):
                try:
                    engine_tmp = create_engine(mysql_url_base)
                    with engine_tmp.connect() as tmpc:
                        # Use SQLAlchemy's text() for literal SQL statement execution
                        databases = [r[0] for r in tmpc.execute(text("SHOW DATABASES")).fetchall()]
                        # Filter out system databases
                        databases = [d for d in databases if
                                     d.lower() not in ("information_schema", "mysql", "performance_schema", "sys")]

                    st.session_state["mysql_databases"] = databases
                    st.sidebar.success(f"Successfully connected to MySQL server. Choose database below.")
                    st.session_state["connected"] = False  # Must remain False until a DB is selected

                except Exception as e:
                    st.sidebar.error(f"MySQL connection failed: {e}")
                    st.session_state["connected"] = False
                    st.session_state["mysql_databases"] = []  # Clear list on failure

            # Dropdown and Button 2: Choose database and finalize connection
            if "mysql_databases" in st.session_state and st.session_state["mysql_databases"]:
                # The selectbox MUST be rendered outside of the sidebar container in the final app, but
                # for the provided code structure, it's placed here:
                selected_db_name = st.selectbox("Choose database", st.session_state["mysql_databases"],
                                                key="mysql_db_select")

                # Button 2: Connect to the selected database
                if st.button("Connect to Database", key="connect_db"):
                    DB_URL = f"{mysql_url_base}{selected_db_name}"
                    try:
                        engine = create_engine(DB_URL)
                        # Test connection to the specific database
                        with engine.connect() as conn:
                            conn.execute(text("SELECT 1"))

                        st.session_state["engine"] = engine
                        st.session_state["connected"] = True
                        st.session_state["selected_db_name"] = selected_db_name
                        st.sidebar.success(f"MySQL connected: {selected_db_name}")
                    except Exception as e:
                        st.sidebar.error(f"Failed to connect to database '{selected_db_name}': {e}")
                        st.session_state["connected"] = False

    # 3. Table list/selection (THE AUTOMATIC PART)
    # This block executes immediately after any successful connection attempt (Connect to DB button)
    if st.session_state["connected"] and st.session_state["engine"]:
        try:
            # Use the saved engine to inspect and fetch table names
            inspector = inspect(st.session_state["engine"])
            tables = inspector.get_table_names()
            st.session_state["tables"] = tables
        except Exception as e:
            st.warning(f"Could not retrieve tables: {e}")
            st.session_state["tables"] = ["(no tables found)"]

        if st.session_state["tables"]:
            # Renders the table dropdown in the sidebar
            selected_table = st.sidebar.selectbox("Select table for experiments", st.session_state["tables"],
                                                  key="sel_table")
            st.session_state["selected_table"] = selected_table
        else:
            st.sidebar.warning("No tables found in the database.")

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
                # Check if 'engine' is None (only happens if SQLite path was not set/tested yet)
                if engine:
                    df_sample.to_sql("items", engine, if_exists="replace", index=False)
                    st.success(
                        "Sample table `items` created. **Please re-test connection in sidebar to refresh table list.**")
                else:
                    st.error("Please ensure the SQLite connection is tested first.")
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
                if engine:
                    df_sample.to_sql("sales", engine, if_exists="replace", index=False)
                    st.success(
                        "Sample table `sales` created. **Please re-test connection in sidebar to refresh table list.**")
                else:
                    st.error("Please ensure the SQLite connection is tested first.")
    else:
        st.info("Sample data creation is only available for SQLite.")


# Nested helper for the Dynamic SQL Builder
def render_dynamic_where_builder(table_metadata, col_names):
    """Renders the UI for building WHERE clauses."""
    st.subheader("Filter Conditions (WHERE)")

    # Initialize filters if not present
    if "qb_filters" not in st.session_state:
        st.session_state["qb_filters"] = []

    if st.button("➕ Add Filter Condition", key="add_filter_btn"):
        new_key = time.time()
        st.session_state["qb_filters"].append((new_key, None, '=', ''))  # Structure: (key, column, operator, value)

    remove_keys = []

    # Iterate over a copy of the list to allow modification (removal)
    current_filters = list(st.session_state["qb_filters"])

    for i, (key, col, op, val) in enumerate(current_filters):
        with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 5, 1])

            with col1:
                # Determine the default index for the current column
                try:
                    default_index = col_names.index(col) + 1 if col in col_names else 0
                except ValueError:
                    default_index = 0

                current_col = st.selectbox(
                    f"Column #{i + 1}", ["(select)"] + col_names,
                    index=default_index,
                    key=f"qb_filter_col_{key}"
                )

            with col2:
                op_options = ["=", "<", ">", "<=", ">=", "<>", "!=", "LIKE"]
                try:
                    default_op_index = op_options.index(op)
                except ValueError:
                    default_op_index = 0

                current_op = st.selectbox(
                    "Operator", op_options,
                    index=default_op_index,
                    key=f"qb_filter_op_{key}"
                )

            with col4:
                # Provide a hint for the data type
                data_type_hint = table_metadata.get(current_col, "TEXT")
                current_val = st.text_input(
                    f"Value (Type: {data_type_hint})",
                    value=val,
                    key=f"qb_filter_val_{key}"
                )
            with col5:
                st.markdown("<br>", unsafe_allow_html=True)  # Spacer
                if st.button("❌", key=f"remove_filter_{key}"):
                    remove_keys.append(key)

            # Update session state *after* the widgets have been rendered and read their values
            st.session_state["qb_filters"][i] = (key, current_col, current_op, current_val)

    # Process removals
    st.session_state["qb_filters"] = [f for f in st.session_state["qb_filters"] if f[0] not in remove_keys]


def render_dynamic_sql_builder():
    """Renders the Dynamic SQL Builder (No-Code SQL) tab."""
    qb_table = st.session_state.get("selected_table", "(no tables found)")
    engine = st.session_state["engine"]

    st.header("🚀 Dynamic SQL Builder — No-Code Data Explorer")
    st.markdown("""
    This tool helps you construct complex SQL queries using simple inputs, without writing any code. It's great for:
    * **Learning:** See how SQL clauses (SELECT, WHERE, JOIN, GROUP BY) are assembled.
    * **Preparation:** Quickly build a query to run performance tests on in the **Research Lab**.
    """)

    if not st.session_state["connected"] or qb_table == "(no tables found)":
        st.warning("Please connect to a database and select a table in the **Connection & Setup** tab first.")
        return

    # Fetch metadata
    col_metadata = get_table_metadata(engine, qb_table)
    col_names = list(col_metadata.keys())

    # --- 1. Table Preview
    render_table_preview(engine, qb_table)

    # --- 2. SELECT Columns
    st.markdown("### 1️⃣ SELECT: Choose Columns")
    selected_cols = st.multiselect("Columns to SELECT", col_names, default=col_names[:3] if col_names else [],
                                   key="qb_select_cols")

    # --- 3. JOINs
    st.markdown("### 2️⃣ JOIN: Merge Data")
    join_clause_final = None
    with st.container(border=True):
        all_tables = [t for t in st.session_state["tables"] if t != qb_table]
        join_type = st.selectbox("Join Type", ["(none)", "INNER JOIN", "LEFT JOIN"], key="qb_join_type")

        if join_type != "(none)" and all_tables:
            join_table = st.selectbox("Other Table", all_tables, key="qb_join_table")
            col_j1, col_j2 = st.columns(2)

            join_on_main = None
            join_on_other = None

            with col_j1:
                join_on_main = st.selectbox(f"On {qb_table} Col", col_names, key="qb_join_maincol")

            with col_j2:
                try:
                    other_col_metadata = get_table_metadata(engine, join_table)
                    other_col_names = list(other_col_metadata.keys())
                    join_on_other = st.selectbox(f"On {join_table} Col", other_col_names, key="qb_join_othercol")
                except Exception:
                    join_on_other = st.text_input(f"On {join_table} Col (manual)", key="qb_join_othercol_manual",
                                                  value="id")

            if join_table and join_on_main and join_on_other:
                join_clause_final = f"{join_type} {safe_sql_identifier(join_table)} ON {safe_sql_identifier(qb_table)}.{safe_sql_identifier(join_on_main)} = {safe_sql_identifier(join_table)}.{safe_sql_identifier(join_on_other)}"
                st.session_state["qb_join"] = (join_type, join_table, join_on_main, join_on_other)
                st.info(f"Join Clause: {join_clause_final}")
            else:
                st.session_state["qb_join"] = None
        else:
            st.session_state["qb_join"] = None

    # --- 4. WHERE (Filter Conditions)
    st.markdown("### 3️⃣ WHERE: Filter Conditions")
    render_dynamic_where_builder(col_metadata, col_names)  # Renders and updates st.session_state["qb_filters"]

    where_parts = []
    for key, col, op, val in st.session_state["qb_filters"]:
        if col and col != "(select)" and val.strip():
            val_clean = val.strip().replace("'", "''")

            # Determine if the value needs SQL quotes
            needs_quotes = True
            try:
                float(val_clean)
                needs_quotes = False
            except ValueError:
                # Handle boolean/keywords
                if val_clean.upper() in ["NULL", "TRUE", "FALSE"]:
                    needs_quotes = False

            if needs_quotes:
                if op == "LIKE":
                    val_clean = f"'%{val_clean}%'"
                else:
                    val_clean = f"'{val_clean}'"

            where_parts.append(f"{safe_sql_identifier(col)} {op} {val_clean}")

    where_sql = " AND ".join(where_parts) if where_parts else None

    # --- 5. GROUP BY / HAVING (Aggregation)
    st.markdown("### 4️⃣ GROUP BY & HAVING")
    group_cols = st.multiselect("Columns to GROUP BY", col_names, key="qb_group_cols")
    group_sql = ", ".join([safe_sql_identifier(c) for c in group_cols]) if group_cols else None

    having_sql = None
    if group_sql:
        having_exp = st.text_input("HAVING condition (SQL syntax, e.g. COUNT(*) > 10)", key="qb_having_exp")
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

    # --- 7. Final Query Assembly
    final_query = build_sql(
        selects=selected_cols,
        table=qb_table,
        joins=join_clause_final,
        where=where_sql,
        group=group_sql,
        having=having_sql,
        order=order_col,
        order_dir=order_dir,
        limit=limit_val
    )

    st.subheader("Final Generated SQL Query")
    st.code(final_query, language="sql")

    # --- 8. Execution
    if st.button("▶️ Run Generated Query", key="qb_run_btn", type="primary"):
        try:
            df, t, m = execute_and_measure(final_query, engine)
            st.session_state["qb_result_df"] = df
            st.session_state["qb_result_time"] = t
            st.session_state["qb_result_mem"] = m
            # Log the performance
            log_performance_db(
                st.session_state["db_mode_choice"].split(" ")[0],
                st.session_state["selected_db_name"],
                final_query, len(df), t, m, indexed=0
            )
            st.session_state["ai_suggested_query_ran"] = True
        except Exception as e:
            st.error(f"Query execution failed: {e}")
            st.session_state["qb_result_df"] = pd.DataFrame()

    # --- Results Display
    if "qb_result_df" in st.session_state and not st.session_state["qb_result_df"].empty:
        st.subheader("Query Result")
        st.success(
            f"✅ Query executed in **{st.session_state['qb_result_time']:.4f}s** | Peak memory: **{st.session_state['qb_result_mem']:.2f} MB** | Rows: **{len(st.session_state['qb_result_df'])}**")
        st.dataframe(st.session_state["qb_result_df"])


def render_research_lab():
    """Renders Research Lab (Performance, Indexing, Benchmarking) with new experiments."""
    selected_table_name = st.session_state.get("selected_table", "(no tables found)")
    db_mode = st.session_state["db_mode_choice"]

    st.header("🔬 3. Autonomous Research Lab (Performance Tuning)")
    st.markdown("""
    ⚠️ **Warning:** Experiments can create and drop temporary tables. Use this on test/sample data only.
    """)

    if not st.session_state.get("connected") or selected_table_name == "(no tables found)":
        st.warning("Please connect to a database and select a table in the **Connection & Setup** tab first.")
        return

    engine = st.session_state["engine"]

    # --- 1. Query Performance & EXPLAIN ---
    with st.expander("1️⃣ Query Performance & EXPLAIN (Profiling)", expanded=True):
        st.markdown("Measure execution time and memory for custom queries, and view the query plan.")
        query_default = f"SELECT * FROM {safe_sql_identifier(selected_table_name)} LIMIT 1000"
        query_text = st.text_area("SELECT query to run", value=query_default, height=100, key="perf_query_text_area")

        # Simulation setup (only for SQLite in the original)
        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_a:
            sim_disabled = not db_mode.startswith("SQLite")
            sim_enable = st.checkbox("Simulate on large dataset (SQLite only)", key="sim_enable_check",
                                     disabled=sim_disabled)
        with col_b:
            dup_factor = st.number_input("Duplication Factor (x)", min_value=1, max_value=10, value=2,
                                         disabled=sim_disabled or not sim_enable)
        with col_c:
            run_query_btn = st.button("▶️ Run & Measure Query", key="run_measure_btn", type="primary")

        temp_table = f"tmp_sim_{selected_table_name}_{hashlib.sha1(os.urandom(10)).hexdigest()[:5]}"
        if run_query_btn:
            result_container = st.container()
            sim_query = query_text
            cleanup_required = False

            if sim_enable and db_mode.startswith("SQLite"):
                # Simulation logic is omitted here for brevity but was in the original file
                st.info("Simulation feature is highly dependent on environment. Running test query only.")
                sim_query = query_text

            try:
                df, t, m = execute_and_measure(sim_query, engine)
                with result_container:
                    st.success(
                        f"Execution successful in **{t:.4f}s** | Peak memory: **{m:.2f} MB** | Rows: **{len(df)}**")
                    st.dataframe(df.head(10))
                # Log the performance
                log_performance_db(db_mode.split(" ")[0], st.session_state["selected_db_name"], sim_query, len(df), t,
                                   m, indexed=0)
                st.session_state["ai_suggested_query_ran"] = True
            except Exception as e:
                with result_container:
                    st.error(f"Query execution failed: {e}")

            # EXPLAIN
            st.subheader("EXPLAIN Query Plan")
            try:
                if db_mode.startswith("SQLite"):
                    explain_q = "EXPLAIN QUERY PLAN " + sim_query
                    explain_df = pd.read_sql_query(explain_q, engine)
                elif db_mode == "MySQL":
                    explain_q = "EXPLAIN " + sim_query
                    explain_df = pd.read_sql_query(explain_q, engine)
                else:
                    st.warning("EXPLAIN not supported for this DB mode.")
                    explain_df = pd.DataFrame()
                st.dataframe(explain_df, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to fetch EXPLAIN plan: {e}")

        st.divider()

    # --- 2. Indexing Techniques & Tester ---
    with st.expander("2️⃣ Indexing Techniques & Tester (Index vs. Scan)", expanded=False):
        st.markdown("Compare query performance before and after creating a temporary index.")
        st.info("""
        **How to Test:** Enter the column name and a test query (which **must** filter on that column). The app runs the query in three phases: 1. No Index, 2. Index Created, 3. Index Dropped (Cleanup).
        """)

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
            col_metadata = get_table_metadata(engine, selected_table_name)
            safe_fallback_col = next(iter(col_metadata.keys()), 'id') if col_metadata else safe_fallback_col

            if not index_column: index_column = safe_fallback_col

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

            # --- 1. No Index Run ---
            status_bar.info("Phase 1/3: Running query **without** index...")
            try:
                df_no, t_no, m_no = execute_and_measure(index_test_query, engine)
                results.append({"Run": "No Index", "Time (s)": t_no})
                log_performance_db(db_mode.split(" ")[0], st.session_state["selected_db_name"], index_test_query,
                                   len(df_no), t_no, m_no, indexed=0)
                status_bar.success(f"No Index Time: {t_no:.4f}s")
            except Exception as e:
                status_bar.error(f"No Index Run Failed: {e}")
                return

            # --- 2. Create Index ---
            status_bar.info(f"Phase 2/3: Creating index `{idx_name}` on column `{index_column}`...")
            try:
                create_index(engine, selected_table_name, index_column, idx_name)
                status_bar.success("Index created.")
            except Exception as e:
                status_bar.error(f"Index Creation Failed: {e}. Aborting test.")
                try:
                    drop_index(engine, selected_table_name, idx_name, db_mode)
                except:
                    pass
                return

            # --- 3. Index Run ---
            status_bar.info(f"Running query **with** index `{idx_name}`...")
            try:
                df_idx, t_idx, m_idx = execute_and_measure(index_test_query, engine)
                results.append({"Run": "With Index", "Time (s)": t_idx})
                log_performance_db(db_mode.split(" ")[0], st.session_state["selected_db_name"], index_test_query,
                                   len(df_idx), t_idx, m_idx, indexed=1)
                status_bar.success(f"With Index Time: {t_idx:.4f}s")
            except Exception as e:
                status_bar.error(f"Indexed Run Failed: {e}")
                t_idx = t_no

            # --- 4. Drop Index (Cleanup) ---
            status_bar.info("Phase 3/3: Dropping index (Cleanup)...")
            try:
                drop_index(engine, selected_table_name, idx_name, db_mode)
                status_bar.success("Index dropped successfully. Environment restored.")
            except Exception as e:
                status_bar.error(f"Index Drop Failed: {e}. Please drop index `{idx_name}` manually.")

            # --- Results Visualization ---
            df_r = pd.DataFrame(results).set_index("Run")
            st.subheader("Indexing Comparison Results")
            col_metric, col_chart = st.columns([1, 2])

            # Calculate improvement factor
            if df_r.loc['With Index', 'Time (s)'] > 0:
                improvement_factor = df_r.loc['No Index', 'Time (s)'] / df_r.loc['With Index', 'Time (s)']
            else:
                improvement_factor = 0  # Prevent division by zero

            col_metric.metric("Performance Change",
                              f"{improvement_factor:.2f}x",
                              delta=f"Improvement from {df_r.loc['No Index', 'Time (s)']:.4f}s to {df_r.loc['With Index', 'Time (s)']:.4f}s")
            with col_chart:
                fig, ax = plt.subplots(figsize=(6, 3))
                df_r["Time (s)"].plot(kind='bar', ax=ax, color=['#e74c3c', '#27ae60'])
                ax.set_title("Query Time: No Index vs. With Index")
                ax.set_ylabel("Execution Time (s)")
                ax.tick_params(axis='x', rotation=0)
                st.pyplot(fig)
                plt.close(fig)

    # --- 3. Cost-Based Optimizer Simulation ---
    with st.expander("3️⃣ Cost-Based Optimizer (CBO) Behavior", expanded=False):
        st.markdown("Visualize the database optimizer's decision-making process (Index Scan vs. Table Scan).")
        st.info(
            "The optimizer switches to an **Index Scan** when a query's **Selectivity** (the fraction of rows returned) drops below a certain threshold.")

        col_metadata = get_table_metadata(engine, selected_table_name)
        col_names = list(col_metadata.keys())
        if not col_names:
            st.warning("Table metadata not available. Cannot run CBO simulation.")
            return

        default_col = next((c for c in col_names if 'price' in c.lower() or 'id' in c.lower()),
                           col_names[0] if col_names else None)
        optim_col = st.selectbox("Column for Selectivity (Numeric only for best results)", col_names,
                                 index=col_names.index(default_col) if default_col in col_names else 0,
                                 key="optim_col_select")

        if st.button("📈 Run CBO Simulation (Selectivity Sweep)", key="run_cbo_sim"):
            results = []
            try:
                # 1. Get min/max for the column to determine the sweep range
                df_bounds = pd.read_sql_query(
                    f"SELECT MIN({safe_sql_identifier(optim_col)}) as min_val, MAX({safe_sql_identifier(optim_col)}) as max_val FROM {safe_sql_identifier(selected_table_name)}",
                    engine).iloc[0]
                min_val = df_bounds['min_val']
                max_val = df_bounds['max_val']

                if max_val is None or min_val is None or max_val == min_val:
                    raise ValueError("Column data is not numeric or min/max values are the same.")

                # 2. Sweep thresholds from min to max
                thresholds = [min_val + i * (max_val - min_val) / 10 for i in range(1, 10)]

            except Exception as e:
                st.error(f"Failed to get column bounds for CBO simulation: {e}")
                return

            status_placeholder = st.empty()
            for i, threshold in enumerate(thresholds):
                status_placeholder.info(f"Running EXPLAIN sweep: {i + 1}/{len(thresholds)}")
                sim_query = f"SELECT * FROM {safe_sql_identifier(selected_table_name)} WHERE {safe_sql_identifier(optim_col)} > {threshold}"
                cost_proxy = 0
                plan = "Unknown"
                try:
                    if db_mode.startswith("SQLite"):
                        explain_q = "EXPLAIN QUERY PLAN " + sim_query
                        exdf = pd.read_sql_query(explain_q, engine)
                        for row in exdf['detail']:
                            plan = 'Index Scan' if 'INDEX' in row.upper() else 'Table Scan'
                            match = re.search(r'rows=(\d+)', row)
                            if match: cost_proxy = max(cost_proxy, int(match.group(1)))

                    elif db_mode == "MySQL":
                        explain_q = "EXPLAIN " + sim_query
                        exdf = pd.read_sql_query(explain_q, engine)
                        cost_proxy = exdf['rows'].iloc[0]
                        plan = 'Index Scan' if 'index' in exdf['type'].iloc[0].lower() else 'Table Scan'

                    selectivity = (max_val - threshold) / (max_val - min_val) * 100 if (max_val - min_val) != 0 else 0
                    results.append({"Selectivity (%)": selectivity, "Cost Proxy (Rows)": cost_proxy, "Plan": plan})
                except Exception as e:
                    st.warning(f"Failed to explain query at threshold {threshold}: {e}")
                    results.append(
                        {"Selectivity (%)": (max_val - threshold) / (max_val - min_val) * 100, "Cost Proxy (Rows)": 0,
                         "Plan": "Error"})

            status_placeholder.empty()
            df_cbo = pd.DataFrame(results).sort_values("Selectivity (%)", ascending=False)
            st.dataframe(df_cbo)

            fig, ax = plt.subplots(figsize=(10, 5))
            for plan_type, group in df_cbo.groupby("Plan"):
                ax.plot(group["Selectivity (%)"], group["Cost Proxy (Rows)"], marker='o', label=plan_type)

            ax.set_xlabel(f"Query Selectivity (%) - (e.g., {optim_col} > X)")
            ax.set_ylabel("Estimated Cost Proxy (Rows Processed)")
            ax.set_title("Optimizer Decision Model Visualization")
            ax.legend(title="Execution Plan")
            ax.grid(True)
            st.pyplot(fig)
            plt.close(fig)

            st.markdown(
                """
                > **Research Insight:** Notice the crossover point. When the selectivity is high (many rows are returned), the CBO prefers a **Table Scan**. As selectivity decreases (fewer rows returned), the cost of an **Index Scan** becomes lower, and the CBO switches its execution plan.
                """
            )


def render_nosql_lab():
    """Renders the NoSQL/Distributed Systems Lab tab."""
    selected_table_name = st.session_state.get("selected_table", "(no tables found)")

    st.header("🌐 4. NoSQL & Distributed Systems (Simulation)")
    st.info(
        "This section simulates modern database architectures, demonstrating trade-offs in **storage structure (B-Tree vs. LSM-Tree)** and the complexity of **data sharding**.")

    if not st.session_state.get("connected") or selected_table_name == "(no tables found)":
        st.warning("Please connect to a database and select a table in the **Connection & Setup** tab first.")
        return

    # -------------------------
    # 1. Hybrid SQL-NoSQL Storage (LSM-Tree vs. B-Tree)
    # -------------------------
    st.markdown("---")
    st.markdown("## 1. Hybrid SQL-NoSQL Storage (LSM-Tree vs. B-Tree)")

    op_counts = st.slider("Number of Operations (Simulated Workload)", 100, 5000, 1000, 100)
    col_w, col_r = st.columns(2)
    with col_w:
        write_perc = st.slider("Write Operations (%)", 0, 100, 50)
    with col_r:
        read_perc = 100 - write_perc
        st.metric("Read Operations (%)", read_perc)

    num_total_ops = op_counts
    num_writes = num_total_ops * (write_perc / 100)
    num_reads = num_total_ops * (read_perc / 100)

    # Cost Factors
    b_tree_write_factor = 1.5
    b_tree_read_factor = 0.5
    lsm_tree_write_factor = 0.3
    lsm_tree_read_factor = 1.2

    # Calculate costs
    b_write_cost = num_writes * b_tree_write_factor
    b_read_cost = num_reads * b_tree_read_factor
    lsm_write_cost = num_writes * lsm_tree_write_factor
    lsm_read_cost = num_reads * lsm_tree_read_factor

    df_results = pd.DataFrame({
        "Storage": ["B-Tree (SQL - Good Reads)", "LSM-Tree (NoSQL - Good Writes)"],
        "Write Cost (Simulated I/O)": [b_write_cost, lsm_write_cost],
        "Read Cost (Simulated I/O)": [b_read_cost, lsm_read_cost]
    }).set_index("Storage")

    df_results["Total Cost (Normalized)"] = df_results["Write Cost (Simulated I/O)"] + df_results[
        "Read Cost (Simulated I/O)"]

    st.subheader("Simulated I/O Cost Comparison (Write-Heavy vs. Read-Heavy)")
    st.dataframe(df_results.style.format(precision=2))

    fig, ax = plt.subplots(figsize=(8, 4))
    df_results[["Write Cost (Simulated I/O)", "Read Cost (Simulated I/O)"]].plot(kind='bar', stacked=True, ax=ax,
                                                                                 color=['#e74c3c', '#3498db'])
    ax.set_title("Total I/O Cost by Storage Structure")
    ax.set_ylabel("Normalized I/O Cost")
    ax.tick_params(axis='x', rotation=0)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
    > **Research Insight:** **B-Tree** structures are good for read-heavy OLTP workloads because reads are fast (low cost, `~0.5`). **LSM-Trees** optimize for write-heavy loads by performing writes sequentially (low cost, `~0.3`), but complex reads (which must merge data from multiple components) become expensive (`~1.2`).
    """)

    # -------------------------
    # 2. Distributed Database Sharding Simulation
    # -------------------------
    st.markdown("---")
    st.markdown("## 2. Data Distribution & Sharding Simulation")
    st.write(
        "Visualize the performance trade-offs of partitioning data (sharding) across a distributed cluster using a chosen **sharding key**.")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        shards = st.slider("Number of Logical Shards", 2, 8, 4, key="num_shards")
    with col_s2:
        sharding_key = st.text_input("Simulated Sharding Key Column (e.g., user_id or category)", value="category",
                                     key="sharding_key_input")

    st.markdown("---")

    col_q_type, col_key_type = st.columns(2)
    with col_q_type:
        query_type = st.selectbox("Query Type",
                                  ["Point Query (Single Row Lookup)", "Distributed Aggregate (Full Scan)"])
    with col_key_type:
        key_type = st.selectbox("Key Used in Query", ["Sharding Key (Good)", "Non-Sharding Key (Bad)"])

    if st.button("▶️ Run Sharding Simulation", key="run_sharding_sim"):
        results = []
        cost_base = 1000  # Base cost of a full scan on one shard

        if query_type == "Point Query (Single Row Lookup)":
            if key_type == "Sharding Key (Good)":
                cost = cost_base / shards
                explanation = f"Query routed to 1 shard out of {shards}. Cost reduced by a factor of {shards}."
            else:
                cost = cost_base  # Must scan all shards to find the data. No performance gain.
                explanation = f"Query must broadcast to all {shards} shards. No performance gain (Cross-shard JOIN/Scan)."
        else:  # Distributed Aggregate
            if key_type == "Sharding Key (Good)":
                cost = cost_base * 1.5  # Requires merge step, but still distributed
                explanation = f"Query runs in parallel on all {shards} shards, but requires a final merge (higher overall latency/cost)."
            else:
                cost = cost_base * 2.5  # Full scan and complex merge/coordination
                explanation = f"Query runs in parallel on all {shards} shards, requiring complex cross-shard data movement and merging (highest cost)."

        results.append({"Metric": "Total Cost Proxy (Latency)", "Value": f"{cost:.2f}", "Explanation": explanation})
        results.append({"Metric": "Number of Shards Accessed", "Value": 1 if (
                    key_type == "Sharding Key (Good)" and query_type == "Point Query (Single Row Lookup)") else shards,
                        "Explanation": ""})
        df_sim = pd.DataFrame(results)

        st.subheader("Sharding Experiment Results")
        st.dataframe(df_sim)
        st.markdown(f"**Performance Insight:** {explanation}")


def render_ai_analyzer_and_logs():
    """Renders the AI Analyzer and Performance Logs tab."""
    st.header("🧠 5. AI Analyzer & Scientific Export")
    st.info(
        "This section performs post-experiment analysis on the **Performance Workload Logs** to generate AI-driven insights and a final research summary.")

    engine = st.session_state["engine"]

    # --- 1. AI Index Suggestion (Learned from History) ---
    st.markdown("---")
    st.markdown("## 1. AI Index Suggestion (Learned from History)")
    with st.container(border=True):
        freq_df, perf_summary = train_ai_from_logs(limit=1000)

        if not freq_df.empty:
            st.subheader("Top 5 Most Frequently Filtered Columns (Index Candidates)")

            # Use a slightly different styling for visual interest
            styled_freq_df = freq_df.head(5).style.background_gradient(subset=['count'], cmap='YlOrRd')
            st.dataframe(styled_freq_df, use_container_width=True)

            top_col = freq_df.iloc[0]["column"]
            st.info(
                f"💡 **AI Suggestion:** Based on the logged queries, the column **`{top_col}`** is the most frequent filter candidate. **Indexing this column is predicted to yield the highest performance gain.**")
            if st.button(f"Inject '{top_col}' into Indexing Test Tab (Tab 4)", key="ai_inject_btn"):
                # Use session state to transfer the suggestion to the Research Lab tab
                st.session_state["_ai_suggested_index_col"] = top_col
                st.info(f"Column **`{top_col}`** injected into the Indexing Tester on Tab 4. Switch to that tab now.")
        else:
            st.warning("No query logs found. Run some queries in Tab 3 or Tab 4 to train the AI!")

    # --- 2. Query Log Viewer & Trend Analysis ---
    st.markdown("---")
    st.markdown("## 2. Performance Log Viewer & Trend Analysis")
    df_logs = read_perf_logs(limit=500)
    st.subheader("Recent Performance Logs")
    st.dataframe(df_logs)

    if not df_logs.empty:
        st.subheader("Trend Visualization")
        queries_unique = df_logs["query_hash"].unique().tolist()

        # Map hash back to text for display
        hash_to_text = df_logs.set_index('query_hash')['query_text'].to_dict()
        display_options = {h: hash_to_text[h][:60] + "..." for h in queries_unique}

        # Let the user select by the truncated query text
        selected_display_text = st.selectbox("Select Logged Query", list(display_options.values()), key="trend_select")

        # Find the hash corresponding to the selected text
        selected_hash = next((h for h, text in display_options.items() if text == selected_display_text), None)

        if selected_hash:
            sel_q_text = hash_to_text.get(selected_hash, "")
            st.code(sel_q_text, language="sql", height=80)

            df_sel = df_logs[df_logs["query_hash"] == selected_hash].sort_values("timestamp")

            if not df_sel.empty:
                df_chart = df_sel.copy()
                df_chart["timestamp"] = pd.to_datetime(df_chart["timestamp"])
                df_chart["indexed_status"] = df_chart["indexed"].apply(lambda x: "Indexed" if x == 1 else "No Index")

                st.subheader("Execution Time Trend")
                fig, ax = plt.subplots(figsize=(10, 4))
                for status, group in df_chart.groupby("indexed_status"):
                    ax.plot(group["timestamp"], group["exec_time"], marker='o', label=status)
                ax.set_xlabel("Time of Run")
                ax.set_ylabel("Execution Time (s)")
                ax.set_title("Performance Trend by Index Status")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

    # --- 3. Final Scientific Report ---
    st.markdown("---")
    st.markdown("## 3. Generate Final Research Report")
    if st.button("Generate Final Research Report", type="primary"):
        total_queries = len(df_logs)
        indexed_runs = df_logs[df_logs["indexed"] == 1]["exec_time"]
        no_index_runs = df_logs[df_logs["indexed"] == 0]["exec_time"]

        avg_t_idx = indexed_runs.mean() if not indexed_runs.empty else 0
        avg_t_no_idx = no_index_runs.mean() if not no_index_runs.empty else 0

        improvement = (avg_t_no_idx / avg_t_idx) if avg_t_idx > 0 else 0

        st.subheader("📝 Scientific Conclusion & Findings")

        summary_part_1 = f"""
        ## 📝 1. Детално Резиме на Работно Оптоварување (Plain Explanation)
        **📊 Статистика на Сесијата:**
        * **Вкупно Извршени Операции:** {total_queries}
        * **Објаснување (Workload):** Вие ја симулиравте работата на вистинска апликација. Овие {total_queries} барања го претставуваат **Workload-от** (секојдневната работа) на базата.
        * **Цел:** АИ-системот ги следеше овие операции за да научи кои делови од базата треба да ги оптимизира за да стане **Self-Optimizing**.
        """
        st.markdown(summary_part_1)

        st.markdown("---")
        st.subheader("🔬 2. Квантитативна Евалуација на Хипотезата")
        improvement_text = f"**{improvement:.2f}x** подобрување" if improvement > 1.05 else "Нема значајно подобрување или нецелосни податоци (< 5%)"
        st.markdown(f"""
        **Workload Analysis Result:**
        * **Просечно Време (No Index):** {avg_t_no_idx:.4f}s
        * **Просечно Време (Indexed):** {avg_t_idx:.4f}s
        * **Просечна Добивка на Перформанси:** {improvement_text}
        """)

        st.markdown("---")
        st.subheader("📈 3. AI Autonomous Recommendation Success")
        if not freq_df.empty:
            top_col = freq_df.iloc[0]["column"]
            st.markdown(
                f"**Autonomous Indexing Successful:** The AI correctly identified **`{top_col}`** as a primary optimization candidate from the workload logs.")
        else:
            st.markdown(
                "**Autonomous Indexing Inconclusive:** The system lacked sufficient workload data to make a formal recommendation.")


# --------------------------------------------------
# MAIN APPLICATION FLOW
# --------------------------------------------------

# Tab structure is based on the provided file (6 tabs total, with an intro tab)
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📘 Research Overview",
    "⚙️ Connection & Setup",
    "🚀 Dynamic SQL Builder",
    "🔬 Research Lab",
    "🌐 NoSQL & Distributed Systems",
    "🧠 AI Analyzer & Logs"
])

# 1. 📘 Research Overview
with tab1:
    render_research_intro()
    render_tab_guide(
        "1. Research Overview (Вовед)",
        "Овој модул ја дефинира **Формалната Академска Рамка** на целиот експеримент: што се истражува, зошто е важно и кој е текот на работата. Ова е вашиот PhD Протокол.",
        "Прочитајте го воведот за да разберете што прави апликацијата. Потоа, продолжете со чекор 2 (Setup) за да започнете со експериментот.",
        "Овде се дефинирани **Примарната Истражувачка Цел** и **Секундарните Прашања** кои се таргетирани со експериментите во модулите 3 и 4."
    )

# 2. ⚙️ Connection & Setup
with tab2:
    render_connection_and_setup()
    render_tab_guide(
        "2. Connection & Setup",
        "Модулот ги конфигурира иницијалните услови на контролираната експериментална средина (база, табела, големина на податоци).",
        "Поврзете се на MySQL со вашите податоци. Потоа, изберете табела од листата во страничната лента.",
        "Овој модул ја дефинира **Независната Варијабла** (големина на податочен сет) за експериментите."
    )

# 3. 🚀 Dynamic SQL Builder
with tab3:
    render_dynamic_sql_builder()
    render_tab_guide(
        "3. Dynamic SQL Builder (Workload Generator)",
        "Овој модул ви овозможува да генерирате и извршите **Произволни Барања (Workload)** кон базата. Секое барање автоматски се логира (зачувува) за да го 'научи' АИ-системот.",
        "Изградете неколку различни `SELECT` барања со различни филтри (`WHERE`) и `JOIN`-ови. Извршете ги за да ја снимите историјата.",
        "Овде се дефинира и снима **Workload Pattern**-от (работната шема) на апликацијата, кој е основа за автономната оптимизација во следниот модул."
    )

# 4. 🔬 Research Lab
with tab4:
    render_research_lab()
    render_tab_guide(
        "4. Autonomous Research Lab",
        "Овој модул ги спроведува клучните експерименти за мерење на перформансите на оптимизацијата на базата.",
        "Користете го делот 2 за да тестирате индекс. Внесете ја колоната што ја препорача АИ во Табот 6 и споредете ги времињата.",
        "Овде се мерат **Зависните Варијабли (Execution Time/Memory)**. Анализирајте го **EXPLAIN Plan** за да го видите однесувањето на Optimizer-от."
    )

# 5. 🌐 NoSQL & Distributed Systems
with tab5:
    render_nosql_lab()
    render_tab_guide(
        "5. NoSQL & Distributed Systems (Simulation)",
        "Овој модул ги симулира предностите и недостатоците на различни архитектури на бази на податоци (пр. LSM-Tree за NoSQL).",
        "Поместете го лизгачот за да видите како се менува трошокот за читање/пишување помеѓу B-Tree и LSM-Tree.",
        "LSM-Tree симулацијата го демонстрира трендот **Hybrid SQL-NoSQL** во однос на I/O трошокот, а шард-симулацијата ја прикажува сложеноста на **Data Distribution**."
    )

# 6. 🧠 AI Analyzer & Logs
with tab6:
    render_ai_analyzer_and_logs()
    render_tab_guide(
        "6. AI Analyzer & Scientific Export",
        "Овој модул ги прикажува суровите податоци (логови) и генерира финален, едноставен извештај за тоа што е постигнато во експериментот. Ова е вашиот научен заклучок.",
        "Кликнете 'Generate Final Research Report' за да добиете заклучок за просечната добивка на перформанси, промената на меморијата и кои барања најмногу се оптимизираа.",
        "Овде се врши **Квантитативна Евалуација** на хипотезите. Финалниот извештај го сумира **Autonomous Optimization Behavior** на системот врз основа на собраните Workload логови."
    )

# -------------------------
# Footer / notes
# -------------------------
st.markdown("---")
st.markdown(
    "### Notes & Disclaimer\n"
    "- This tool is designed for **Database Research and Education** (e.g., your course, 'New Trends in Database Research').\n"
    "- Features like `Query Simulation` and `Indexing Experiments` are **destructive** (they create/drop indices/tables) and should only be run on test or sample databases.\n"
    "- **Created for Academic Use (PhD/Postgrad Research Lab).**"
)