import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io


def blob_to_np_array(blob):
    bio = io.BytesIO(blob)
    return np.load(bio)


def inspect_and_plot(db_path: str, output_dir='./plots', max_days=2):
    os.makedirs(output_dir, exist_ok=True)

    conn = sqlite3.connect(db_path)
    print("Connected to DB. Inspecting tables...")

    days_df = pd.read_sql_query("SELECT * FROM Days", conn)
    windows_df = pd.read_sql_query("SELECT * FROM Windows", conn)  # Main data source
    trials_df = pd.read_sql_query("SELECT * FROM Trials", conn)

    print("\nDays Table Summary:")
    print(days_df.head())
    print(f"Total days: {len(days_df)}")

    print("\nWindows Table Summary (first 5 rows):")
    print(windows_df.head())
    print(f"Total window entries: {len(windows_df)}")

    if not windows_df.empty:
        sample_blob = windows_df['blob'].iloc[0]
        sample_array = blob_to_np_array(sample_blob)
        print("\nTest: Deserialized sample window array:")
        print(sample_array[:5])  # First 5 elements
        print(f"Shape: {sample_array.shape}")
    else:
        print("No windows data found. Exiting.")
        conn.close()
        return

    days = sorted(days_df['DID'].unique())[:max_days]
    num_days = len(days)
    popevents_day = {}

    for did in days:
        day_windows = windows_df[windows_df['DID'] == did].copy()
        if day_windows.empty:
            print(f"No data for DID {did}. Skipping.")
            continue

        day_windows['array'] = day_windows['blob'].apply(blob_to_np_array)

        grouped = day_windows.groupby('NID')['array'].apply(lambda x: np.mean(np.stack(x), axis=0))
        popevents = np.stack(grouped.values)

        if popevents.size == 0:
            print(f"No valid popevents for DID {did}. Skipping.")
            continue

        popevents_day[did] = popevents
        print(f"Processed DID {did}: {popevents.shape[0]} neurons, {popevents.shape[1]} frames")

    conn.close()

    if not popevents_day:
        print("No valid data across days. Exiting.")
        return

    sns.set_style('white')
    cmax = .4
    cmin = -cmax

    if num_days > 1:
        fig, axs = plt.subplots(2, num_days, figsize=(7, 8))
    else:
        fig, axs = plt.subplots(2, 1, figsize=(5, 8))

    axs = axs.reshape(2, -1)

    pre_window_size = 50
    for col, did in enumerate(days):
        if did not in popevents_day:
            continue

        temp = popevents_day[did]

        temp_response = np.nanmean(temp, axis=1)
        sort_idx = np.argsort(temp_response)[::-1]
        sorted_temp = temp[sort_idx]
        num_neurons = sorted_temp.shape[0]

        ax = axs[0, col]
        im = ax.imshow(sorted_temp, cmap=plt.get_cmap('PRGn_r'), vmin=cmin, vmax=cmax, aspect='auto')
        ax.grid(False)
        ax.set_ylabel(f'{num_neurons} neurons')
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(f'Day {did}')
        ax.plot([pre_window_size, pre_window_size], [0, num_neurons], '--k', linewidth=1.5)

        ax = axs[1, col]
        mean_trace = np.nanmean(sorted_temp, axis=0)
        ax.plot(mean_trace)
        ax.plot([pre_window_size, pre_window_size], [np.min(mean_trace) - 0.5, np.max(mean_trace) + 0.5], '--k',
                linewidth=1.5)
        ax.plot([0, temp.shape[1]], [0, 0], '--k', linewidth=0.5)
        ax.set_xticks([])
        ax.set_xlabel('Frames')
        ax.set_ylim(-0.15, 0.4)

    fig.tight_layout()
    plt.colorbar(im, ax=axs[0, -1], label='ΔF/F')

    # Save plots
    # plot_path_pdf = os.path.join(output_dir, 'population_heatmaps.pdf')
    # plot_path_png = os.path.join(output_dir, 'population_heatmaps.png')
    # fig.savefig(plot_path_pdf, format='PDF')
    # fig.savefig(plot_path_png, format='PNG')
    # print(f"Plots saved to {plot_path_pdf} and {plot_path_png}")

    plt.show()
    plt.close()  # Close to avoid display in console


if __name__ == "__main__":
    inspect_and_plot(db_path="./output/database.db")