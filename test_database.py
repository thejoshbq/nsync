import sqlite3
import pandas as pd
import numpy as np
import io


# SQLite adapters for NumPy arrays
def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("BLOB", convert_array)


def test_database(db_path='./database/PFC_Self-Admin.db', target_did=100):
    # Connect to the database
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = conn.cursor()

    print("=== Database Information ===")
    # Get all table names
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cur.fetchall()]
    print(f"Tables in database: {tables}")

    # Print schema and row counts for each table
    for table in tables:
        print(f"\n--- Table: {table} ---")
        # Get schema
        cur.execute(f"PRAGMA table_info({table})")
        schema = pd.DataFrame(cur.fetchall(), columns=['cid', 'name', 'type', 'notnull', 'dflt_value', 'pk'])
        print("Schema:")
        print(schema[['name', 'type', 'pk']])
        # Get row count
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        row_count = cur.fetchone()[0]
        print(f"Row count: {row_count}")
        # Get sample data (first 5 rows)
        df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 5", conn)
        # Handle BLOBs by showing shape
        for col in df.columns:
            if df[col].dtype == object and any(isinstance(x, np.ndarray) for x in df[col].dropna()):
                df[col] = df[col].apply(lambda x: f"Array(shape={x.shape})" if isinstance(x, np.ndarray) else x)
        print("Sample data (first 5 rows):")
        print(df)

    print("\n=== Population Analysis for DID = {} ===".format(target_did))
    # Query population data (mimic NSyncPopulation for a day)
    query = """
            SELECT w.NID, w.FID, w.Blob, f.DID
            FROM Windows w
                     JOIN FOVs f ON w.FID = f.FID
            WHERE f.DID = ? \
            """
    windows_df = pd.read_sql_query(query, conn, params=(target_did,))

    if windows_df.empty:
        print(f"No data found for DID = {target_did}")
    else:
        # Aggregate windows by neuron (mean across trials)
        windows_by_neuron = []
        for nid in windows_df['NID'].unique():
            neuron_windows = windows_df[windows_df['NID'] == nid]['Blob'].values
            if neuron_windows.size > 0:
                # Stack windows (list of arrays) and compute mean across trials
                neuron_windows = np.stack(neuron_windows, axis=0)  # Shape: (num_trials, window_size)
                mean_window = np.nanmean(neuron_windows, axis=0)  # Shape: (window_size,)
                windows_by_neuron.append(mean_window)

        if windows_by_neuron:
            # Stack all neuron means
            per_neuron_means = np.stack(windows_by_neuron, axis=0)  # Shape: (num_neurons, window_size)
            # Sort by mean response (mimicking NSyncPopulation.sorted_indices)
            mean_responses = np.nanmean(per_neuron_means, axis=1)
            sort_indices = np.argsort(mean_responses)[::-1]
            per_neuron_means = per_neuron_means[sort_indices]

            print(f"Number of neurons for DID = {target_did}: {per_neuron_means.shape[0]}")
            print(f"Window size: {per_neuron_means.shape[1]}")
            print("Mean response per neuron (first 5 neurons):")
            print(mean_responses[sort_indices][:5])

            # Optional: Save per_neuron_means for plotting or further analysis
            np.save('per_neuron_means_did_{}.npy'.format(target_did), per_neuron_means)
        else:
            print(f"No valid windows for DID = {target_did}")

    # Close connection
    conn.close()


if __name__ == "__main__":
    test_database(db_path='./database/PFC_Self-Admin.db', target_did=100)
