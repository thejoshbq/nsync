# System modules
import os
import exdir as ex
import warnings
warnings.filterwarnings('always', category=UserWarning)
warnings.filterwarnings('always', category=DeprecationWarning)

# Data processing modules
import numpy as np
from nsync2p import NSyncSample, NSyncPopulation  # Adjust import based on your module name

# Visualization modules
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    sns.set_style('white')

    root = "./data"
    output_path = os.path.join("./", "compiled_data_2")
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    if os.path.isdir(root):
        days = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
        day_datasets = []
        day_num_neurons_list = []

        for day in days:
            day_path = os.path.join(root, day)
            day_group_name = day.replace(" ", "_")
            day_samples = []
            day_num_neurons = 0

            animals = [a for a in sorted(os.listdir(day_path)) if os.path.isdir(os.path.join(day_path, a))]
            for animal in animals:
                animal_path = os.path.join(day_path, animal)
                animal_group_name = animal.replace(" ", "_")

                fovs = [f for f in sorted(os.listdir(animal_path)) if os.path.isdir(os.path.join(animal_path, f))]
                for fov in fovs:
                    fov_path = os.path.join(animal_path, fov)
                    fov_group_name = fov.replace(" ", "_")

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
                            target_event_id=[22, 222], # active and active-timeout lever presses
                            isolated_events=True,
                            min_events=2,
                            normalize_data=True,
                        )
                        dataset_windows = dataset.get_event_windows()
                        if dataset_windows.ndim == 3 and dataset_windows.size > 0:
                            day_num_neurons += dataset.get_num_neurons()
                            day_samples.append(dataset)
                    except Exception as e:
                        print(f"Error processing {fov_path}: {e}")
                        continue

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

        num_days = len(days)
        fig, axes = plt.subplots(2, num_days, figsize=(15, 8), sharex='col', squeeze=False)
        fig.suptitle("Multi-Day Neural Activity", color='white', fontsize=16)

        for idx, (day_dataset, day, num_valid_neurons) in enumerate(zip(day_datasets, days, day_num_neurons_list)):
            if num_valid_neurons == 0:
                continue

            per_neuron_means = day_dataset.per_neuron_means[day_dataset.sorted_indices()]

            ax1 = axes[0, idx]
            im = ax1.imshow(
                per_neuron_means,
                cmap=plt.get_cmap('PRGn_r'),
                vmin=-4, vmax=4,
                aspect='auto'
            )
            ax1.set_title(f'{day} (n={num_valid_neurons})', color='white')
            ax1.axvline(x=day_dataset.pre_window_size, color='k', linestyle='--', alpha=0.7, label='Event')
            ax1.tick_params(colors='white')
            if idx == 0:
                ax1.set_ylabel('Neurons', color='white')

            ax2 = axes[1, idx]
            ax2.set_ylim(-.5, 2.5)
            ax2.plot(np.nanmean(per_neuron_means, axis=0), color='r', label='Mean Activity')
            ax2.axvline(x=day_dataset.pre_window_size, color='k', linestyle='--', alpha=0.7, label='Event')
            ax2.set_xlabel('Frames Relative to Event', color='white')
            ax2.set_ylabel('Mean Response', color='white')
            ax2.grid(True, color='gray', alpha=0.3)
            ax2.tick_params(colors='white')

        plt.tight_layout()
        plt.show()