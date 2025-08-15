# System modules
import os
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from nsync2p import NSyncSample, NSyncPopulation
import scipy.io as sio
import sqlite3
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

if __name__ == "__main__":
    warnings.filterwarnings('always', category=UserWarning)
    warnings.filterwarnings('always', category=DeprecationWarning)

    # Initialize lists for population data and tables
    day_datasets = []
    day_num_neurons_list = []
    days_table = []
    animals_table = []
    fovs_table = []
    extracted_signals_table = []
    trials_table = []
    windows_table = []

    root = "./data"
    output_path = os.path.join("./", "database")
    os.makedirs(output_path, exist_ok=True)

    if os.path.isdir(root):
        days = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]

        for d, day in tqdm(enumerate(days[:2]), total=len(days), desc="Processing days"):
            day_path = os.path.join(root, day)
            day_group_name = day.replace(" ", "_")
            day_samples = []

            did = d + 100
            days_table.append({"DID": did, "Day": day[2:]})

            animals = [a for a in sorted(os.listdir(day_path)) if os.path.isdir(os.path.join(day_path, a))]
            for a, animal in enumerate(animals):
                animal_path = os.path.join(day_path, animal)
                aid = a + 100
                animals_table.append({"AID": aid, "Animal": animal})

                fovs = [f for f in sorted(os.listdir(animal_path)) if os.path.isdir(os.path.join(animal_path, f))]
                for f, fov in enumerate(fovs):
                    fov_path = os.path.join(animal_path, fov)
                    fid = f + 100
                    fovs_table.append({"FID": fid, "FOV": fov, "AID": aid, "DID": did})

                    extracted_signals_files = sorted([
                        os.path.join(fov_path, f) for f in os.listdir(fov_path)
                        if f.endswith(".npy") and "extractedsignals_raw" in f
                    ])
                    behavior_event_log_files = sorted([
                        os.path.join(fov_path, f) for f in os.listdir(fov_path)
                        if f.endswith(".mat")
                    ])
                    mat_files_no_frames = sorted([
                        os.path.join(fov_path, f) for f in os.listdir(fov_path)
                        if f.endswith(".mat") and "noframes" in f
                    ])

                    try:
                        dataset = NSyncSample(
                            eventlog=behavior_event_log_files,
                            extracted_signals=extracted_signals_files,
                            mat_no_frames=mat_files_no_frames if mat_files_no_frames else None,
                            animal_name=animal,
                            fov=fov,
                            day=day[2:],
                            target_event_id=[22, 222],
                            isolated_events=True,
                            min_events=2,
                            normalize_data=True,
                        )

                        extracted_signals = dataset.get_extracted_signals()
                        eventlog = dataset.get_eventlog()
                        dataset_event_windows = dataset.get_event_windows()

                        # Vectorized trial ID collection
                        if eventlog.size > 0:
                            num_events = eventlog.shape[0]
                            tids = np.arange(10000, 10000 + num_events)
                            trials_table.extend({
                                "TID": tid,
                                "EID": event[0],
                                "Timestamp": event[1],
                                "FID": fid
                            } for tid, event in zip(tids, eventlog))

                        # Vectorized window and neuron ID collection
                        if dataset_event_windows.ndim == 3 and dataset_event_windows.size > 0:
                            day_samples.append(dataset)
                            num_neurons, window_size, num_trials = dataset_event_windows.shape

                            # Generate TIDs and NIDs
                            tids = np.arange(10000, 10000 + num_trials)
                            nids = np.arange(10000, 10000 + num_neurons)

                            # Create meshgrid for TID-NID pairs
                            tid_grid, nid_grid = np.meshgrid(tids, nids, indexing='ij')
                            tid_flat = tid_grid.ravel()
                            nid_flat = nid_grid.ravel()
                            fid_array = np.full_like(tid_flat, fid)
                            window_size_array = np.full_like(tid_flat, window_size)

                            # Flatten windows for each neuron-trial pair
                            windows_flat = dataset_event_windows.transpose(2, 0, 1).reshape(-1, window_size)

                            # Batch append to windows_table
                            windows_table.extend({
                                "NID": nid,
                                "TID": tid,
                                "FID": fid,
                                "Window Size": ws,
                                "blob": window
                            } for nid, tid, fid, ws, window in zip(
                                nid_flat, tid_flat, fid_array, window_size_array, windows_flat))

                            # Vectorized extracted signals collection
                            extracted_signals = dataset.get_extracted_signals()
                            num_frames = extracted_signals.shape[1]
                            fid_array = np.full(num_neurons, fid)
                            extracted_signals_table.extend({
                                "NID": nid,
                                "FID": fid,
                                "blob": signal
                            } for nid, fid, signal in zip(nids, fid_array, extracted_signals))

                    except Exception as e:
                        print(f"Error processing {fov_path}: {e}")

            if not day_samples:
                print(f"No valid samples for {day}; skipping.")
                continue

            day_dataset = NSyncPopulation(day_samples)
            per_neuron_means, mean_responses, num_valid_neurons = day_dataset.valid_trials()
            per_neuron_means = per_neuron_means[day_dataset.sorted_indices()]
            day_datasets.append(day_dataset)
            day_num_neurons_list.append(num_valid_neurons)

            if num_valid_neurons == 0:
                print(f"No valid neurons for {day}; skipping in final plot.")

        if not day_datasets:
            print("No valid datasets for any day; exiting.")
            exit()

        # Convert to DataFrames for verification
        days_table = pd.DataFrame(days_table)
        animals_table = pd.DataFrame(animals_table)
        fovs_table = pd.DataFrame(fovs_table)
        extracted_signals_table = pd.DataFrame(extracted_signals_table)
        trials_table = pd.DataFrame(trials_table)
        windows_table = pd.DataFrame(windows_table)

        # Create/connect to the database
        conn = sqlite3.connect(os.path.join(output_path, 'PFC_Self-Admin.db'), detect_types=sqlite3.PARSE_DECLTYPES)
        cur = conn.cursor()

        # Create tables
        cur.execute('''
        CREATE TABLE IF NOT EXISTS Days (
            DID INTEGER PRIMARY KEY,
            Day TEXT
        )
        ''')

        cur.execute('''
        CREATE TABLE IF NOT EXISTS Animals (
            AID INTEGER PRIMARY KEY,
            Animal TEXT
        )
        ''')

        cur.execute('''
        CREATE TABLE IF NOT EXISTS FOVs (
            FID INTEGER PRIMARY KEY,
            FOV TEXT,
            AID INTEGER,
            DID INTEGER,
            FOREIGN KEY (AID) REFERENCES Animals(AID),
            FOREIGN KEY (DID) REFERENCES Days(DID)
        )
        ''')

        cur.execute('''
        CREATE TABLE IF NOT EXISTS Trials (
            TID INTEGER PRIMARY KEY,
            EID INTEGER,
            Timestamp REAL,
            FID INTEGER,
            FOREIGN KEY (FID) REFERENCES FOVs(FID)
        )
        ''')

        cur.execute('''
        CREATE TABLE IF NOT EXISTS Neurons (
            NID INTEGER PRIMARY KEY,
            FID INTEGER,
            FOREIGN KEY (FID) REFERENCES FOVs(FID)
        )
        ''')

        cur.execute('''
        CREATE TABLE IF NOT EXISTS Windows (
            WID INTEGER PRIMARY KEY AUTOINCREMENT,
            NID INTEGER,
            TID INTEGER,
            FID INTEGER,
            Window_Size INTEGER,
            Blob BLOB,
            FOREIGN KEY (NID) REFERENCES Neurons(NID),
            FOREIGN KEY (TID) REFERENCES Trials(TID),
            FOREIGN KEY (FID) REFERENCES FOVs(FID)
        )
        ''')

        cur.execute('''
        CREATE TABLE IF NOT EXISTS ExtractedSignals (
            NID INTEGER,
            FID INTEGER,
            Blob BLOB,
            PRIMARY KEY (NID, FID),
            FOREIGN KEY (NID) REFERENCES Neurons(NID),
            FOREIGN KEY (FID) REFERENCES FOVs(FID)
        )
        ''')

        # Insert data into tables
        for _, row in days_table.iterrows():
            cur.execute("INSERT OR REPLACE INTO Days (DID, Day) VALUES (?, ?)",
                        (row['DID'], row['Day']))

        for _, row in animals_table.iterrows():
            cur.execute("INSERT OR REPLACE INTO Animals (AID, Animal) VALUES (?, ?)",
                        (row['AID'], row['Animal']))

        for _, row in fovs_table.iterrows():
            cur.execute("INSERT OR REPLACE INTO FOVs (FID, FOV, AID, DID) VALUES (?, ?, ?, ?)",
                        (row['FID'], row['FOV'], row['AID'], row['DID']))

        for _, row in trials_table.iterrows():
            cur.execute("INSERT OR REPLACE INTO Trials (TID, EID, Timestamp, FID) VALUES (?, ?, ?, ?)",
                        (row['TID'], row['EID'], row['Timestamp'], row['FID']))

        # Insert unique neurons from windows_table and extracted_signals_table
        neurons_df = pd.concat([
            windows_table[['NID', 'FID']].drop_duplicates(),
            extracted_signals_table[['NID', 'FID']].drop_duplicates()
        ]).drop_duplicates()
        for _, row in neurons_df.iterrows():
            cur.execute("INSERT OR REPLACE INTO Neurons (NID, FID) VALUES (?, ?)",
                        (row['NID'], row['FID']))

        for _, row in windows_table.iterrows():
            cur.execute("INSERT INTO Windows (NID, TID, FID, Window_Size, Blob) VALUES (?, ?, ?, ?, ?)",
                        (row['NID'], row['TID'], row['FID'], row['Window Size'], row['blob']))

        for _, row in extracted_signals_table.iterrows():
            cur.execute("INSERT OR REPLACE INTO ExtractedSignals (NID, FID, Blob) VALUES (?, ?, ?)",
                        (row['NID'], row['FID'], row['blob']))

        # Commit and close
        conn.commit()
        conn.close()

        # Print DataFrames for verification
        print("Days Table:")
        print(days_table)
        print("\nAnimals Table:")
        print(animals_table)
        print("\nFOVs Table:")
        print(fovs_table)
        print("\nTrials Table:")
        print(trials_table)
        print("\nWindows Table:")
        print(windows_table)
        print("\nExtracted Signals Table:")
        print(extracted_signals_table)