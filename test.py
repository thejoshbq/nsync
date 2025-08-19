import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

def blob_to_array(blob):
    bio = io.BytesIO(blob)
    return np.load(bio)

def process_day_data(conn, did):
    frame_rate = 30
    frame_averaging = 4
    sampling_rate = frame_rate / frame_averaging
    baseline_range = np.arange(int(3 * sampling_rate))

    windows_df = pd.read_sql_query("""
                                   SELECT w.neuron_id, w.blob
                                   FROM EventWindows w
                                            JOIN FOV f ON w.fov_id = f.fov_id
                                   WHERE f.day_id = ?
                                   """, conn, params=(did,))

    if windows_df.empty:
        return None, None, 0

    windows_df['array'] = windows_df['blob'].apply(blob_to_array)

    grouped = windows_df.groupby('neuron_id')['array'].apply(
        lambda x: np.nanmean(np.stack(x), axis=0) if len(x) > 0 else None
    ).dropna()

    if grouped.empty:
        return None, None, 0

    per_neuron_means = np.stack(grouped.values)

    baseline = np.nanmean(per_neuron_means[:, baseline_range], axis=1)[:, np.newaxis]
    per_neuron_means -= baseline

    for i in range(per_neuron_means.shape[0]):
        baseline_std = np.nanstd(per_neuron_means[i, baseline_range])
        if baseline_std > 0:
            per_neuron_means[i] /= baseline_std

    mean_responses = np.nanmean(per_neuron_means, axis=1)

    valid_mask = ~np.isnan(mean_responses)
    if not np.any(valid_mask):
        return None, None, 0

    per_neuron_means = per_neuron_means[valid_mask]
    mean_responses = mean_responses[valid_mask]
    num_valid_neurons = per_neuron_means.shape[0]

    sort_idx = np.argsort(mean_responses)[::-1]
    sorted_temp = per_neuron_means[sort_idx]

    mean_trace = np.nanmean(sorted_temp, axis=0)

    return sorted_temp, mean_trace, num_valid_neurons

if __name__ == "__main__":
    db_path = "./output/PFC_Self-Admin.db"
    conn = sqlite3.connect(db_path)

    days_df = pd.read_sql_query("SELECT day_id, label FROM Days ORDER BY label", conn)
    days = days_df['label'].tolist()
    day_id = days_df['day_id'].tolist()
    num_days = len(days)

    if num_days == 0:
        print("No days found in database.")
        conn.close()
        exit()

    sns.set_style('white')

    fig, axes = plt.subplots(2, num_days, figsize=(15, 8), sharex='col', squeeze=False)
    fig.suptitle("Multi-Day Neural Activity from Database\n\n", fontsize=16)

    for idx, (did, day) in enumerate(zip(day_id, days)):
        sorted_temp, mean_trace, num_valid_neurons = process_day_data(conn, did)

        if num_valid_neurons == 0:
            axes[0, idx].set_title(f'{day} (n=0)')
            axes[1, idx].set_title('No data')
            continue

        ax1 = axes[0, idx]
        im = ax1.imshow(
            sorted_temp,
            cmap=plt.get_cmap('PRGn_r'),
            vmin=-4, vmax=4,
            aspect='auto'
        )
        ax1.set_title(f'{day[2:]} (n={num_valid_neurons})')
        ax1.axvline(x=75, color='k', linestyle='--', alpha=0.7)  # pre_window_size=75
        ax1.tick_params(labelbottom=False)

        ax2 = axes[1, idx]
        ax2.plot(mean_trace, color='r')
        ax2.axvline(x=75, color='k', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Frames')
        ax2.set_ylim(-0.5, 2.5)
        ax2.grid(True, color='gray', alpha=0.3)

        if idx == 0:
            ax1.set_ylabel('Neurons')
            ax2.set_ylabel('Mean Response')

    plt.colorbar(im, ax=axes[0, -1], label='Z-scored ΔF/F')
    plt.tight_layout()
    plt.show()

    conn.close()