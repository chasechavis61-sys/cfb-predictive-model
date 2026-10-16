import os
import sqlite3
import pandas as pd
import streamlit as st
import cfbd

# -----------------------------
# 1. Load CFBD API Key
# -----------------------------

def load_cfbd_key():
    """
    Loads CFBD API key from environment or SQLite database.
    """
    # First try environment variable
    if "BEARER_TOKEN" in os.environ:
        return os.environ["BEARER_TOKEN"]

    # Otherwise load from DB
    conn = sqlite3.connect("cfb.db")
    row = conn.execute(
        "SELECT api_key FROM api_keys WHERE service='cfbd'"
    ).fetchone()
    conn.close()

    if row:
        os.environ["BEARER_TOKEN"] = row[0]
        return row[0]

    return None


# -----------------------------
# 2. Fetch Weekly Rankings
# -----------------------------

def fetch_weekly_rankings(year):
    """
    Calls CFBD API and returns weekly rankings as a list of dicts.
    """
    key = load_cfbd_key()
    if not key:
        raise ValueError("CFBD API key not found. Add it to DB or environment.")

    config = cfbd.Configuration(access_token=key)

    with cfbd.ApiClient(config) as api_client:
        api = cfbd.RankingsApi(api_client)
        rankings = api.get_rankings(year=year)

    return [r.to_dict() for r in rankings]


# -----------------------------
# 3. Write Rankings to Database
# -----------------------------

def write_rankings_to_db(rows):
    """
    Writes rankings into SQLite table cfb_rankings_weekly.
    """
    conn = sqlite3.connect("cfb.db")
    df = pd.DataFrame(rows)
    df.to_sql("cfb_rankings_weekly", conn, if_exists="append", index=False)
    conn.close()


# -----------------------------
# 4. Streamlit UI
# -----------------------------

st.title("🏈 CFBD Weekly Auto‑Updater")

st.write("This tool fetches weekly CFBD rankings and stores them in your database.")

year = st.number_input("Season Year", min_value=2000, max_value=2030, value=2026)

if st.button("Fetch & Update Weekly Rankings"):
    try:
        st.write("Fetching rankings...")
        rows = fetch_weekly_rankings(year)

        st.write(f"Fetched {len(rows)} ranking entries. Writing to database...")
        write_rankings_to_db(rows)

        st.success("✅ Weekly rankings updated successfully!")

    except Exception as e:
        st.error(f"Error: {e}")


# -----------------------------
# 5. View Data
# -----------------------------

if st.button("View Latest Rankings in DB"):
    conn = sqlite3.connect("cfb.db")
    df = pd.read_sql_query("SELECT * FROM cfb_rankings_weekly ORDER BY week DESC LIMIT 50", conn)
    conn.close()

    st.dataframe(df)
