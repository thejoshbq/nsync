import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nsync2p.sample import NSyncSample
from nsync2p.population import NSyncPopulation
from tqdm import tqdm

def compile_data(root: str = "./", dataframe: bool = False, plot: bool = False):
    if not os.path.isdir(root):
        print(f"Directory {root} does not exist.")
        return

    days = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
    day_datasets = []
    day_num_neurons_list = []

    for day in days:
        day_path = os.path.join(root, day)
        day_samples = []
        day_num_neurons = 0

        animals = [a for a in sorted(os.listdir(day_path)) if os.path.isdir(os.path.join(day_path, a))]
        for animal in animals:
            animal_path = os.path.join(day_path, animal)

            fovs = [f for f in sorted(os.listdir(animal_path)) if os.path.isdir(os.path.join(animal_path, f))]
            for fov in fovs:
                fov_path = os.path.join(animal_path, fov)

                # for f in os.listdir(fov_path):
                #     if "extractedsignals_raw" not in f and not f.endswith(".npz") and not f.endswith(".mat"):
                #         os.remove(os.path.join(fov_path, f))

                extracted_signals_files = sorted([
                    os.path.join(fov_path, f) for f in os.listdir(fov_path)
                    if f.endswith(".npy") and "extractedsignals_raw" in f
                ])
                behavior_event_log_files = sorted([
                    os.path.join(fov_path, f) for f in os.listdir(fov_path)
                    if f.endswith(".mat")
                ])

                try:
                    dataset = NSyncSample(
                        eventlog=behavior_event_log_files,
                        extracted_signals=extracted_signals_files,
                        frame_correction=os.path.join(root, 'empty.mat'),
                        animal_name=animal,
                        target_id=[22, 222], # active and active-timeout lever presses
                        isolate_events=True,
                        min_events=3,
                        normalize=True,
                    )
                    dataset_windows = dataset.get_event_windows()
                    if dataset_windows.ndim == 3 and dataset_windows.size > 0:
                        day_num_neurons += dataset.get_num_neurons()
                        day_samples.append(dataset)
                except Exception as e:
                    print(f"Error processing {fov_path}: {e}")
                    continue

        day_dataset = NSyncPopulation(
            day_samples,
            subtract_baseline=True,
            z_score=True,
            compute_significance=True,
            bh_correction=False,
        )
        per_neuron_means, mean_responses, num_valid_neurons = day_dataset.get_valid_trials()
        day_datasets.append(day_dataset)
        day_num_neurons_list.append(num_valid_neurons)

        if dataframe:
            print(day_dataset)
            print()

    if not day_datasets:
        exit()

    if plot:
        num_days = len(days)
        fig, axes = plt.subplots(2, num_days, figsize=(15, 8), sharex='col', squeeze=False)
        fig.suptitle("Multi-Day Neural Activity", color='k', fontsize=16)

        for idx, (day_dataset, day, num_valid_neurons) in enumerate(zip(day_datasets, days, day_num_neurons_list)):
            if num_valid_neurons == 0:
                continue

            per_neuron_means = day_dataset.per_neuron_means[day_dataset.get_sorted_indices()]

            ax1 = axes[0, idx]
            im = ax1.imshow(
                per_neuron_means,
                cmap=plt.get_cmap('PRGn_r'),
                vmin=-4, vmax=4,
                aspect='auto'
            )
            ax1.set_title(f'{day} (n={num_valid_neurons})', color='k')
            ax1.axvline(x=day_dataset.get_pre_window_size(), color='k', linestyle='--', alpha=0.7, label='Event')
            ax1.tick_params(colors='k')

            ax2 = axes[1, idx]
            ax2.set_ylim(-.5, 2.5)
            ax2.plot(np.nanmean(per_neuron_means, axis=0), color='r', label='Mean Activity')
            ax2.axvline(x=day_dataset.get_pre_window_size(), color='k', linestyle='--', alpha=0.7, label='Event')
            ax2.set_xlabel('Frames Relative to Event', color='k')
            ax2.grid(True, color='gray', alpha=0.3)
            ax2.tick_params(colors='k')

            if idx == 0:
                ax1.set_ylabel('Neurons', color='k')
                ax2.set_ylabel('Mean Response', color='k')

        plt.colorbar(im, ax=axes[0, -1], label='Z-scored ΔF/F')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    sns.set_style('white')
    compile_data("../data", False, True)
