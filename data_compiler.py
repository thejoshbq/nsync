# System modules
import os
import exdir as ex
import warnings
warnings.filterwarnings('always', category=UserWarning)
warnings.filterwarnings('always', category=DeprecationWarning)

# Data processing modules
import numpy as np
import NSync2P as nsync

# Visualization modules
import matplotlib.pyplot as plt

root = "./data"
output_path = os.path.join("./", "compiled_data_2")
if not os.path.exists(output_path):
    os.mkdir(output_path)

if os.path.isdir(root):
    for day in sorted(os.listdir(root)):
        day_path = os.path.join(root, day)
        if not os.path.isdir(day_path):
            continue

        day_group_name = day.replace(" ", "_")
        day_group_dataset = []
        day_num_neurons = 0

        for animal in sorted(os.listdir(day_path)):
            animal_path = os.path.join(day_path, animal)
            if not os.path.isdir(animal_path):
                continue

            animal_group_name = animal.replace(" ", "_")

            for fov in sorted(os.listdir(animal_path)):
                fov_path = os.path.join(animal_path, fov)
                if not os.path.isdir(fov_path):
                    continue

                fov_group_name = fov.replace(" ", "_")

                extracted_signals_files = sorted([os.path.join(fov_path, f) for f in os.listdir(fov_path) if f.endswith(".npy") and "extractedsignals_raw" in f])
                behavior_event_log_files = sorted([os.path.join(fov_path, f) for f in os.listdir(fov_path) if f.endswith(".mat")])

                dataset = nsync.NSyncTimeline(
                    eventlog=behavior_event_log_files,
                    extracted_signals=extracted_signals_files,
                    target_event_id=[22, 222],
                    isolated_events=True,
                    min_events=1
                )
                dataset_windows = dataset.event_windows()
                if dataset_windows.shape[0] != 0 and dataset_windows.size != 0:
                    day_num_neurons += dataset.num_neurons()
                    day_group_dataset.append(dataset_windows)

        max_trials = max(w.shape[2] for w in day_group_dataset)
        window_size = day_group_dataset[0].shape[1]
        aligned_windows = []
        for windows in day_group_dataset:
            if windows.shape[2] < max_trials:
                padded = np.pad(
                    windows,
                    ((0, 0), (0, 0), (0, max_trials - windows.shape[2])),
                    mode='constant',
                    constant_values=np.nan
                )
            else:
                padded = windows
            aligned_windows.append(padded)

        all_windows = np.concatenate(aligned_windows, axis=0)
        per_neuron_means = np.nanmean(all_windows, axis=2)  # total_neurons x window
        baseline_range = np.arange(0, int(3 * (30 / 40)))  # Match original
        baseline = np.nanmean(per_neuron_means[:, baseline_range], axis=1)[:, None]
        per_neuron_means -= baseline

        z_scored = True  # Configurable
        if z_scored:
            for i in range(per_neuron_means.shape[0]):
                baseline_std = np.nanstd(per_neuron_means[i, baseline_range])
                per_neuron_means[i] /= baseline_std if baseline_std > 0 else 1

        day_group_dataset_mean = np.nanmean(per_neuron_means, axis=0)

        # plot mean activity
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(day_group_dataset_mean, color='blue', label='Mean Activity')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Event')
        ax.set_xlabel('Time Relative to Event (seconds)')
        ax.set_ylabel('Mean Signal Intensity')
        ax.set_title(f'{day} (s={len(os.listdir(day_path))}, n={day_num_neurons}): Mean Neural Activity Across Windows')
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        plt.show()