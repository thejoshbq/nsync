# test.py

import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nsync2p.sample import NSyncSample
from nsync2p.population import NSyncPopulation


def compile_data(root: str = "./", dataframe: bool = False, plot: bool = False, remove_unused_files: bool = False):
    if not os.path.isdir(root):
        print(f"Directory {root} does not exist.")
        return

    cluster_ids = np.load(os.path.join(root, "cluster_ids.npy"))
    used_cluster_ids = 0

    days = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
    day_datasets = []
    day_num_neurons_list = []

    for day in days:
        day_path = os.path.join(root, day)
        day_samples = []

        animals = [a for a in sorted(os.listdir(day_path)) if os.path.isdir(os.path.join(day_path, a))]
        for animal in animals:
            animal_path = os.path.join(day_path, animal)

            fovs = [f for f in sorted(os.listdir(animal_path)) if os.path.isdir(os.path.join(animal_path, f))]
            for fov in fovs:
                fov_path = os.path.join(animal_path, fov)

                if remove_unused_files:
                    for f in os.listdir(fov_path):
                        if "extractedsignals_raw" not in f and not f.endswith(".npz") and not f.endswith(".mat"):
                            os.remove(os.path.join(fov_path, f))

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
                        target_id=[22, 222],  # active and active-timeout lever presses
                        isolate_events=True,
                        min_events=2,
                        normalize=False,
                    )
                    dataset_windows = dataset.get_event_windows()
                    if dataset_windows.ndim == 3 and dataset_windows.size > 0:
                        day_samples.append(dataset)
                    else:
                        print(f"Skipping {fov_path} due to invalid data")
                except Exception as e:
                    print(f"Error processing {fov_path}: {e}")
                    continue

        day_dataset = NSyncPopulation(
            day_samples,
            name=day[2:],
            subtract_baseline=True,
            z_score_baseline=True,
            compute_significance=False,
            bh_correction=False,
        )

        # Compute valid mask before filtering (matches get_valid_trials() logic exactly)
        if day_dataset.per_neuron_means.size > 0:
            mean_responses_pre = np.nanmean(day_dataset.per_neuron_means, axis=1)
            valid_mask_full = ~np.isnan(mean_responses_pre)
            num_valid_check = np.sum(valid_mask_full)
            full_neurons = day_dataset.per_neuron_means.shape[0]
        else:
            valid_mask_full = np.array([], dtype=bool)
            num_valid_check = 0
            full_neurons = 0
        print(f"Day {day}: Full neurons: {full_neurons}, Computed valid: {num_valid_check}")

        per_neuron_means, mean_responses, num_valid_neurons = day_dataset.get_valid_trials()
        print(f"  After filter: {num_valid_neurons}")
        if num_valid_check != num_valid_neurons:
            print("  -> Mismatch! Check preprocessing or data.")

        day_datasets.append(day_dataset)
        day_num_neurons_list.append(num_valid_neurons)

        # Assignment loop: only over used samples, assign only to valid neurons
        if len(day_dataset.get_used_samples()) == 0 or num_valid_neurons == 0:
            print(f"Day {day}: Skipping assignment (no used samples or valid neurons)")
            continue

        cumul_neuron_start = 0
        used_cluster_ids_start = used_cluster_ids
        for sample_idx, sample in enumerate(tqdm(day_dataset.get_used_samples(), desc=f"{day} | Assigning cluster IDs",
                                                 total=len(day_dataset.get_used_samples()))):
            num_sample_neurons = sample.get_num_neurons()
            if num_sample_neurons == 0:
                continue

            sample_neuron_start = cumul_neuron_start
            sample_neuron_end = sample_neuron_start + num_sample_neurons

            # Slice the full valid mask for this sample's neurons
            sample_valid_mask = valid_mask_full[sample_neuron_start:sample_neuron_end]
            num_valid_in_sample = np.sum(sample_valid_mask)

            # Initialize cluster IDs: -1 for invalid neurons
            sample_cluster_ids = np.full(num_sample_neurons, -1, dtype=cluster_ids.dtype)

            # Assign sequential cluster IDs only to valid positions (in local order)
            valid_offset = 0
            for local_idx in range(num_sample_neurons):
                if sample_valid_mask[local_idx]:
                    sample_cluster_ids[local_idx] = cluster_ids[used_cluster_ids + valid_offset]
                    valid_offset += 1

            sample.set_cluster_ids(sample_cluster_ids)

            # Increment used only by valid count
            used_cluster_ids += num_valid_in_sample

            # Advance cumulative for next sample
            cumul_neuron_start = sample_neuron_end

        print(
            f"Day {day}: Assigned {used_cluster_ids - used_cluster_ids_start} valid cluster IDs (total used: {used_cluster_ids}/{len(cluster_ids)})")

        if dataframe:
            print(day_dataset)
            print()

    if not day_datasets:
        exit()

    if plot:
        num_days = len(days)
        min_num_events = day_datasets[0].get_samples()[0].get_min_events()
        fig, axes = plt.subplots(3, num_days, figsize=(15, 12), sharex='col', squeeze=False)
        fig.suptitle(
            f"Normalized Extracted Signals at Event of Interest\n\nEOI: Active Lever Press, n>={min_num_events}\n\n",
            color='k', fontsize=16)

        for idx, (day_dataset, day, num_valid_neurons) in enumerate(zip(day_datasets, days, day_num_neurons_list)):
            if num_valid_neurons == 0:
                continue

            # Compute cluster means before sorting (using unsorted valid per_neuron_means and corresponding cluster IDs)
            start_idx = sum(day_num_neurons_list[:idx])
            day_cluster_ids = cluster_ids[start_idx: start_idx + num_valid_neurons]
            cluster_means = {}
            unique_cids = np.unique(day_cluster_ids)
            for cid in unique_cids:
                mask = (day_cluster_ids == cid)
                if np.sum(mask) > 0:
                    cluster_means[int(cid)] = np.nanmean(day_dataset.per_neuron_means[mask], axis=0)

            # Now sort for imshow
            per_neuron_means = day_dataset.per_neuron_means[day_dataset.get_sorted_indices()]

            ax1 = axes[0, idx]
            im = ax1.imshow(
                per_neuron_means,
                cmap=plt.get_cmap('PRGn_r'),
                vmin=-4, vmax=4,
                aspect='auto'
            )
            ax1.set_title(f'{day[2:]} (n={num_valid_neurons})', color='k')
            ax1.axvline(x=day_dataset.get_pre_window_size(), color='k', linestyle='--', alpha=0.7, label='Event')
            ax1.tick_params(colors='k')

            ax2 = axes[1, idx]
            ax2.set_ylim(-.5, 2.5)
            ax2.plot(np.nanmean(per_neuron_means, axis=0), color="k", label='Mean')
            ax2.plot(np.nanmedian(per_neuron_means, axis=0), color="grey", label='Median')
            ax2.axvline(x=day_dataset.get_pre_window_size(), color='k', linestyle='--', alpha=0.7, label='Event')
            ax2.set_xlabel('Frames Relative to Event', color='k')
            ax2.grid(True, color='gray', alpha=0.3)
            ax2.tick_params(colors='k')
            ax2.set_xticks([0, day_dataset.get_pre_window_size(), len(per_neuron_means[0]) - 1])

            ax3 = axes[2, idx]
            ax3.set_ylim(-8, 12)
            if len(cluster_means) > 0:
                colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_means)))
                for i, (cid, mean_trace) in enumerate(cluster_means.items()):
                    ax3.plot(mean_trace, color=colors[i], label=f'Cluster {cid}')
            ax3.axvline(x=day_dataset.get_pre_window_size(), color='k', linestyle='--', alpha=0.7, label='Event')
            ax3.set_xlabel('Frames Relative to Event', color='k')
            ax3.grid(True, color='gray', alpha=0.3)
            ax3.tick_params(colors='k')
            ax3.set_xticks([0, day_dataset.get_pre_window_size(), len(per_neuron_means[0]) - 1])

            if idx == 0:
                ax1.set_ylabel('Neurons', color='k')
                ax2.set_ylabel('Mean Response', color='k')
                ax3.set_ylabel('Cluster Mean Response', color='k')

        plt.colorbar(im, ax=axes[0, -1], label='Z-scored ΔF/F')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    sns.set_style('white')
    compile_data("data", True, True, False)