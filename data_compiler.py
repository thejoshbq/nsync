# System modules
import os
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from nsync2p import NSyncSample, NSyncPopulation
import sqlite3
import io

def np_array_to_blob(arr):
    """
    Serialize a NumPy array to bytes for storage as a BLOB in SQLite.
    """
    bio = io.BytesIO()
    np.save(bio, arr)
    bio.seek(0)
    return sqlite3.Binary(bio.read())

def compile_data(data: str, root: str=".", output: str="output", name: str="database.db"):
    data_path = os.path.join(root, data)
    output_path = os.path.join(root, output)
    os.makedirs(output_path, exist_ok=True)

    warnings.filterwarnings('always', category=UserWarning)
    warnings.filterwarnings('always', category=DeprecationWarning)

    day_datasets = []
    day_num_neurons_list = []
    days_table = []
    animals_table = []
    fovs_table = []
    extracted_signals_table = []
    trials_table = []
    windows_table = []

    if os.path.isdir(data_path):
        days = [d for d in sorted(os.listdir(data_path)) if os.path.isdir(os.path.join(data_path, d))]

        for d, day in tqdm(enumerate(days), total=len(days), desc="Processing days"):
            day_path = os.path.join(data_path, day)
            day_samples = []

            did = d
            days_table.append({"DID": did, "Day": day[2:]})

            animals = [a for a in sorted(os.listdir(day_path)) if os.path.isdir(os.path.join(day_path, a))]
            for a, animal in enumerate(animals):
                animal_path = os.path.join(day_path, animal)
                aid = a
                animals_table.append({"AID": aid, "Animal": animal})

                fovs = [f for f in sorted(os.listdir(animal_path)) if os.path.isdir(os.path.join(animal_path, f))]
                for f, fov in enumerate(fovs):
                    fov_path = os.path.join(animal_path, fov)
                    fid = f
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

                        eventlog = dataset.get_eventlog()
                        dataset_event_windows = dataset.get_event_windows()

                        if eventlog.size > 0:
                            num_events = eventlog.shape[0]
                            tids = np.arange(num_events, dtype=np.int64)
                            trials_table.extend({
                                "TID": int(tid),
                                "EID": int(event[0]),
                                "Timestamp": float(event[1]),
                                "FID": int(fid),
                                "AID": int(aid),
                                "DID": int(did)
                            } for tid, event in zip(tids, eventlog))

                        if dataset_event_windows.ndim == 3 and dataset_event_windows.size > 0:
                            day_samples.append(dataset)
                            num_neurons, window_size, num_trials = dataset_event_windows.shape

                            tids = np.arange(num_trials, dtype=np.int64)
                            nids = np.arange(num_neurons, dtype=np.int64)

                            tid_grid, nid_grid = np.meshgrid(tids, nids, indexing='ij')
                            tid_flat = tid_grid.ravel()
                            nid_flat = nid_grid.ravel()
                            fid_array = np.full_like(tid_flat, fid, dtype=np.int64)
                            aid_array = np.full_like(tid_flat, aid, dtype=np.int64)
                            did_array = np.full_like(tid_flat, did, dtype=np.int64)
                            window_size_array = np.full_like(tid_flat, window_size, dtype=np.int64)

                            windows_flat = dataset_event_windows.transpose(2, 0, 1).reshape(-1, window_size)

                            windows_table.extend({
                                "NID": int(nid),
                                "TID": int(tid),
                                "FID": int(fid),
                                "AID": int(aid),
                                "DID": int(did),
                                "Window Size": int(ws),
                                "blob": window
                            } for nid, tid, fid, aid, did, ws, window in zip(
                                nid_flat, tid_flat, fid_array, aid_array, did_array, window_size_array, windows_flat))

                            extracted_signals = dataset.get_extracted_signals()
                            fid_array = np.full(num_neurons, fid, dtype=np.int64)
                            aid_array = np.full(num_neurons, aid, dtype=np.int64)
                            did_array = np.full(num_neurons, did, dtype=np.int64)
                            extracted_signals_table.extend({
                                "NID": int(nid),
                                "FID": int(fid),
                                "AID": int(aid),
                                "DID": int(did),
                                "blob": signal
                            } for nid, fid, aid, did, signal in zip(nids, fid_array, aid_array, did_array, extracted_signals))

                    except Exception as e:
                        print(f"Error processing {fov_path}: {e}")

            if not day_samples:
                print(f"No valid samples for {day}; skipping.")
                continue

            day_dataset = NSyncPopulation(day_samples)
            per_neuron_means, mean_responses, num_valid_neurons = day_dataset.valid_trials()
            day_datasets.append(day_dataset)
            day_num_neurons_list.append(num_valid_neurons)

            if num_valid_neurons == 0:
                print(f"No valid neurons for {day}; skipping in final plot.")

        if not day_datasets:
            print("No valid datasets for any day; exiting.")
            exit()

        days_table = pd.DataFrame(days_table).drop_duplicates()
        animals_table = pd.DataFrame(animals_table).drop_duplicates()
        fovs_table = pd.DataFrame(fovs_table).drop_duplicates(subset=['FID', 'AID', 'DID'])
        trials_table = pd.DataFrame(trials_table).drop_duplicates(subset=['TID', 'FID', 'AID', 'DID'])
        extracted_signals_table = pd.DataFrame(extracted_signals_table).drop_duplicates(subset=['NID', 'FID', 'AID', 'DID'])
        windows_table = pd.DataFrame(windows_table).drop_duplicates(subset=['NID', 'TID', 'FID', 'AID', 'DID'])

        extracted_signals_table['blob'] = extracted_signals_table['blob'].apply(np_array_to_blob)
        windows_table['blob'] = windows_table['blob'].apply(np_array_to_blob)

        db_path = os.path.join(output_path, name)
        conn = sqlite3.connect(db_path)
        days_table.to_sql('Days', conn, if_exists='replace', index=False)
        animals_table.to_sql('Animals', conn, if_exists='replace', index=False)
        fovs_table.to_sql('FOVs', conn, if_exists='replace', index=False)
        trials_table.to_sql('Trials', conn, if_exists='replace', index=False)
        extracted_signals_table.to_sql('ExtractedSignals', conn, if_exists='replace', index=False)
        windows_table.to_sql('Windows', conn, if_exists='replace', index=False)
        conn.close()

        print(f"Database created at {db_path}. Compress it (e.g., ZIP) for sharing.")

if __name__ == "__main__":
    compile_data(data="./data", output="./output", name="PFC_Self-Admin.db")