import os
import sqlite3
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nsync2p.sample import NSyncSample

def array_to_blob(arr):
    bio = io.BytesIO()
    np.save(bio, arr)
    return bio.getvalue()

def blob_to_array(blob):
    bio = io.BytesIO(blob)
    return np.load(bio)

def get_or_insert_day(cur, label):
    cur.execute("SELECT day_id FROM Days WHERE label=?", (label,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute("INSERT INTO Days (label) VALUES (?)", (label,))
    return cur.lastrowid

def get_or_insert_animal(cur, label):
    cur.execute("SELECT animal_id FROM Animals WHERE label=?", (label,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute("INSERT INTO Animals (label) VALUES (?)", (label,))
    return cur.lastrowid

def insert_fov(cur, label, animal_id, day_id):
    cur.execute("INSERT INTO FOV (label, animal_id, day_id) VALUES (?,?,?)", (label, animal_id, day_id))
    return cur.lastrowid

def plot_day(conn, day_id, day_label):
    frame_rate = 30
    frame_averaging = 4
    sampling_rate = frame_rate / frame_averaging
    pre_window_size = int(10 * sampling_rate)
    baseline_range = np.arange(int(3 * sampling_rate))

    windows_df = pd.read_sql_query("""
        SELECT w.neuron_id, w.blob
        FROM EventWindows w
        JOIN FOV f ON w.fov_id = f.fov_id
        WHERE f.day_id = ?
    """, conn, params=(day_id,))

    if windows_df.empty:
        print(f"No window data for day {day_label}")
        return

    windows_df['array'] = windows_df['blob'].apply(blob_to_array)

    grouped = windows_df.groupby('neuron_id')['array'].apply(
        lambda x: np.nanmean(np.stack(x), axis=0) if len(x) > 0 else np.full((int(20 * sampling_rate) + 1,), np.nan)
    )
    popevents = np.stack(grouped.values)

    if popevents.size == 0 or np.all(np.isnan(popevents)):
        print(f"No valid population events for day {day_label}")
        return

    def subtract_baseline(per, br):
        bl = np.nanmean(per[:, br], axis=1)[:, np.newaxis]
        return per - bl

    def zscore_data(per, br):
        for i in range(per.shape[0]):
            std = np.nanstd(per[i, br])
            if std > 0:
                per[i] /= std
        return per

    per_neuron_means = zscore_data(subtract_baseline(popevents, baseline_range), baseline_range)
    mean_responses = np.nanmean(per_neuron_means, axis=1)
    sort_idx = np.argsort(mean_responses)[::-1]
    sorted_temp = per_neuron_means[sort_idx]
    num_neurons = sorted_temp.shape[0]

    sns.set_style('white')
    fig, axs = plt.subplots(2, 1, figsize=(5, 8))

    ax = axs[0]
    im = ax.imshow(sorted_temp, cmap=plt.get_cmap('PRGn_r'), vmin=-4, vmax=4, aspect='auto')
    ax.grid(False)
    ax.set_ylabel(f'{num_neurons} neurons')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(f'Day {day_label}')
    ax.plot([pre_window_size, pre_window_size], [0, num_neurons], '--k', linewidth=1.5)

    ax = axs[1]
    mean_trace = np.nanmean(sorted_temp, axis=0)
    ax.plot(mean_trace, color='r')
    ax.plot([pre_window_size, pre_window_size], [np.min(mean_trace) - 0.5, np.max(mean_trace) + 0.5], '--k', linewidth=1.5)
    ax.plot([0, sorted_temp.shape[1]], [0, 0], '--k', linewidth=0.5)
    ax.set_xticks([])
    ax.set_xlabel('Frames')
    ax.set_ylim(-0.5, 2.5)

    fig.tight_layout()
    plt.colorbar(im, ax=axs[0], label='Z-scored ΔF/F')
    plt.show()
    plt.close()

if __name__ == "__main__":
    root = "./data"
    db_path = "./output/PFC_Self-Admin.db"
    os.makedirs("./output", exist_ok=True)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS EventWindows")
    cur.execute("DROP TABLE IF EXISTS Events")
    cur.execute("DROP TABLE IF EXISTS ExtractedSignals")
    cur.execute("DROP TABLE IF EXISTS FOV")
    cur.execute("DROP TABLE IF EXISTS Animals")
    cur.execute("DROP TABLE IF EXISTS Days")
    conn.commit()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS Days (
            day_id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT UNIQUE
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS Animals (
            animal_id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT UNIQUE
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS FOV (
            fov_id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT,
            animal_id INTEGER,
            day_id INTEGER,
            FOREIGN KEY (animal_id) REFERENCES Animals (animal_id),
            FOREIGN KEY (day_id) REFERENCES Days (day_id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ExtractedSignals (
            neuron_id INTEGER PRIMARY KEY AUTOINCREMENT,
            fov_id INTEGER,
            blob BLOB,
            FOREIGN KEY (fov_id) REFERENCES FOV (fov_id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS Events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            type INTEGER,
            timestamp REAL,
            fov_id INTEGER,
            FOREIGN KEY (fov_id) REFERENCES FOV (fov_id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS EventWindows (
            window_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER,
            neuron_id INTEGER,
            fov_id INTEGER,
            blob BLOB,
            FOREIGN KEY (event_id) REFERENCES Events (event_id),
            FOREIGN KEY (neuron_id) REFERENCES ExtractedSignals (neuron_id),
            FOREIGN KEY (fov_id) REFERENCES FOV (fov_id)
        )
    """)
    conn.commit()

    days = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]

    for day in days:
        day_id = get_or_insert_day(cur, day)
        day_path = os.path.join(root, day)
        animals = [a for a in sorted(os.listdir(day_path)) if os.path.isdir(os.path.join(day_path, a))]

        for animal in animals:
            animal_id = get_or_insert_animal(cur, animal)
            animal_path = os.path.join(day_path, animal)
            fovs = [f for f in sorted(os.listdir(animal_path)) if os.path.isdir(os.path.join(animal_path, f))]

            for fov in fovs:
                fov_path = os.path.join(animal_path, fov)
                fov_id = insert_fov(cur, fov, animal_id, day_id)

                extracted_signals_files = sorted([
                    os.path.join(fov_path, ff) for ff in os.listdir(fov_path)
                    if ff.endswith(".npy") and "extractedsignals_raw" in ff
                ])
                behavior_event_log_files = sorted([
                    os.path.join(fov_path, ff) for ff in os.listdir(fov_path)
                    if ff.endswith(".mat")
                ])

                try:
                    dataset = NSyncSample(
                        eventlog=behavior_event_log_files,
                        extracted_signals=extracted_signals_files,
                        frame_correction=os.path.join(root, 'empty.mat'),
                        animal_name=animal,
                        target_id=[22, 222],
                        isolate_events=True,
                        min_events=3,
                        normalize_data=True,
                    )

                    signals = dataset.get_extracted_signals()
                    num_neurons = dataset.get_num_neurons()
                    neuron_ids = []

                    for n in range(num_neurons):
                        signal_blob = array_to_blob(signals[n])
                        cur.execute("INSERT INTO ExtractedSignals (fov_id, blob) VALUES (?, ?)", (fov_id, signal_blob))
                        neuron_ids.append(cur.lastrowid)

                    eventlog = dataset.get_eventlog()
                    for typ, ts in eventlog:
                        cur.execute("INSERT INTO Events (type, timestamp, fov_id) VALUES (?, ?, ?)", (int(typ), float(ts), fov_id))

                    valid_events_ts = dataset.__get_valid_events()
                    target_event_ids = []
                    epsilon = 1e-9
                    for ts in valid_events_ts:
                        cur.execute("""
                            SELECT event_id FROM Events
                            WHERE fov_id = ? AND type IN (22, 222) AND ABS(timestamp - ?) < ?
                        """, (fov_id, ts, epsilon))
                        row = cur.fetchone()
                        if row:
                            target_event_ids.append(row[0])
                        else:
                            print(f"No matching event for timestamp {ts} in FOV {fov}")

                    windows = dataset.__align_event_windows()
                    if windows.ndim != 3 or windows.size == 0:
                        continue

                    num_trials = windows.shape[2]
                    for i in range(num_trials):
                        if i >= len(target_event_ids):
                            break
                        event_id = target_event_ids[i]
                        for n in range(num_neurons):
                            window_arr = windows[n, :, i]
                            if np.all(np.isnan(window_arr)):
                                continue
                            window_blob = array_to_blob(window_arr)
                            neuron_id = neuron_ids[n]
                            cur.execute("INSERT INTO EventWindows (event_id, neuron_id, fov_id, blob) VALUES (?, ?, ?, ?)",
                                        (event_id, neuron_id, fov_id, window_blob))

                except Exception as e:
                    print(f"Error processing {fov_path}: {e}")
                    continue

        conn.commit()
        plot_day(conn, day_id, day)

    conn.close()