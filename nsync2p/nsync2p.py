import os
from pathlib import Path
import scipy.io as sio
from scipy import stats
from sklearn.metrics import roc_auc_score
import pandas as pd
import warnings
warnings.filterwarnings('always', category=UserWarning)
warnings.filterwarnings('always', category=DeprecationWarning)

import numpy as np
from numpy.typing import NDArray
from typing import Any, Dict, List, Union

import matplotlib.pyplot as plt
import seaborn as sns

class NSyncSample:
    def __init__(
        self,
        eventlog: List[Union[str, Path]] | Dict[str, Any],
        extracted_signals: List[Union[str, Path]] | NDArray[np.uint64],
        mat_no_frames: Union[str, Path] | None = None,
        animal_name: str = "REX",
        target_event_id: Union[int, List[int]] = 22,
        noise_event_id: Union[int, List[int]] = 222,
        frame_rate: int = 30,
        frame_averaging: int = 4,
        isolated_events: bool = False,
        min_events: int = 3,
        normalize_data: bool = False
    ):
        self.eventlog = eventlog
        self.extracted_signals = extracted_signals
        self.mat_no_frames = mat_no_frames if mat_no_frames else None
        self.animal_name = animal_name

        if isinstance(eventlog, list):
            self.eventlog = sorted([str(f) for f in eventlog])
            self.eventlog = self._compile_matlab_files(eventlog)
        else:
            self.eventlog = np.array(eventlog)
        self.event_ids = self.eventlog[:, 0]
        self.event_timestamps = self.eventlog[:, 1]

        if isinstance(extracted_signals, list):
            self.extracted_signals = sorted([str(f) for f in extracted_signals])
            self.extracted_signals = self._compile_npy_files(extracted_signals)
        else:
            self.extracted_signals = extracted_signals.astype(np.float64)

        self.num_neurons = self.extracted_signals.shape[0]
        self.num_frames = self.extracted_signals.shape[1]

        # sample processing logic
        self.frame_rate = frame_rate
        self.frame_averaging = frame_averaging
        self.target_event_id = target_event_id
        self.noise_event_id = noise_event_id
        self.min_events = min_events
        self.isolated_events = isolated_events
        self.sampling_rate = frame_rate / frame_averaging
        self.time_per_frame = 1 / frame_rate

        # window of interest
        self.pre_window_size = int(10 * self.sampling_rate)
        self.window_size = int((self.pre_window_size * 2) + (1.6 * self.sampling_rate))
        self.post_window_size = self.window_size - self.pre_window_size

        # baseline and normalization
        self.baseline_range = np.arange(int(3 * self.sampling_rate))
        self.normalize_data = normalize_data
        if self.normalize_data:  # (dF/F)
            self.extracted_signals = self._normalize_signals(self.extracted_signals)

        self.eventlog_dict = { # defined MATLAB event IDs
            "active_lever": 22,
            "active_lever_timeout": 222,
            "inactive_lever": 21,
            "inactive_lever_timeout": 212,
            "cue": 7,
            "infusion": 4,
        }

        if self.mat_no_frames:
            frame_ts = self._get_frame_timestamps(mat_file=self.mat_no_frames, animal=self.animal_name)
        else:
            frame_ts = self._get_frame_timestamps(animal=self.animal_name)
        if self.extracted_signals.shape[1] > frame_ts.shape[0]:
            self.extracted_signals = self.extracted_signals[:, :frame_ts.shape[0] - 1]

    @staticmethod
    def _compile_matlab_files(files: list) -> NDArray[np.float64]:
        stack = []
        last_timestamp = 0
        if isinstance(files, list) and len(files) > 0:
            for file in files:
                try:
                    data = sio.loadmat(file)
                    eventlog = np.squeeze(data["eventlog"])
                    eventlog[:, 1] = eventlog[:, 1] + last_timestamp  # offset timestamps
                    last_timestamp = np.max(eventlog[:, 1])
                    stack.append(eventlog[:, 0:2])
                except Exception as e:
                    print(f"Error loading {file}: {e}")
            stack = np.vstack(stack).squeeze() if len(stack) > 0 else np.squeeze(stack)
            stack = stack[stack[:, 0] != 0]  # only return valid data
        return stack.astype(np.float64)

    @staticmethod
    def _compile_npy_files(files: list) -> NDArray[np.float64]:
        stack = []
        if isinstance(files, list) and len(files) > 0:
            for file in files:
                with warnings.catch_warnings(record=True) as captured_warnings:
                    warnings.simplefilter("always")
                    try:
                        data = np.load(file).squeeze()
                        stack.append(data)
                    except Exception as e:
                        print(f"Error loading {file}: {e}")
                        continue
                    for warning in captured_warnings:
                        if "Python 2" in warning.message:
                            np.save(file, data)
                            print(f"Resaved {file} as a Python 3 file.")
            stack = np.hstack(stack) if len(stack) > 0 else np.array(stack)
        return stack.astype(np.float64)

    @staticmethod
    def _normalize_signals(extracted_signals: NDArray[np.uint64]) -> NDArray[np.float64]:
        means = np.nanmean(extracted_signals, axis=1).reshape(-1, 1)  # shape: (num_neurons, 1)
        normalized_signals = np.divide(
            extracted_signals,
            means,
            out=np.zeros_like(extracted_signals, dtype=np.float64),
            where=(means != 0) & (~np.isnan(means))
        )

        for neuron in range(normalized_signals.shape[0]):
            mean = np.nanmean(normalized_signals[neuron])
            std = np.nanstd(normalized_signals[neuron])
            normalized_signals[neuron] = (normalized_signals[neuron] - mean) / std

        return normalized_signals

    def _get_valid_events(self, sep: float = 1000.0) -> NDArray[np.float64]:
        target_ids = [self.target_event_id] if isinstance(self.target_event_id, int) else self.target_event_id
        events = self.event_timestamps[np.isin(self.event_ids, target_ids)]

        if len(events) == 0:
            print("No target events found")
            return np.array([])

        if self.isolated_events:
            temp = np.sort(events)
            temp = np.delete(temp, np.argwhere(np.diff(temp) < sep) + 1)
            events = temp

        if len(events) < self.min_events:
            print(f"Insufficient valid target events ({len(events)} < {self.min_events})")
            return np.array([])

        return events

    def get_num_events(self) -> int:
        return len(self._get_valid_events())

    def _get_frame_timestamps(self, mat_file: Union[str, Path] | None = None, animal: str = None) -> NDArray[
        np.float64]:
        frame_ts = self.event_timestamps[self.event_ids == 9]

        if frame_ts.size == 0:
            if animal in ['CTL1', 'ER-L1', 'ER-L2', 'IG-19', 'IG-28', 'PGa-T1', 'XYZ'] and mat_file is not None:
                try:
                    assumed_data = sio.loadmat(mat_file)
                    empty_eventlog = np.squeeze(assumed_data['eventlog'])

                    # triplication to extend timestamps
                    max_of = np.max(empty_eventlog[:, 1])
                    length_of = len(empty_eventlog[:, 1])
                    x = np.vstack((empty_eventlog, empty_eventlog, empty_eventlog))
                    x[length_of:, 1] += max_of
                    x[2 * length_of:, 1] += 2 * max_of
                    empty_eventlog = x

                    frame_ts = empty_eventlog[empty_eventlog[:, 0] == 9, 1]

                    dropped_frames = []
                    diff_frames = np.diff(frame_ts)
                    inter_frame_interval = 33
                    frame_drop_idx = np.where(diff_frames > 1.5 * inter_frame_interval)[0]
                    for idx in frame_drop_idx:
                        numframesdropped = int(
                            np.round((frame_ts[idx + 1] - frame_ts[idx]) / (inter_frame_interval + 0.0)) - 1)
                        temp = [frame_ts[idx] + a * inter_frame_interval for a in range(1, numframesdropped + 1)]
                        dropped_frames.extend(temp)
                    corrected = np.sort(np.concatenate((frame_ts, np.array(dropped_frames))))
                    frame_ts = corrected
                except FileNotFoundError:
                    print(f"Assumed timestamps file not found. Generating uniform timestamps based on signal length.")
                    num_frames = self.get_num_frames() * self.frame_averaging
                    frame_ts = np.arange(num_frames) * (1000 / self.frame_rate)

            else:
                print(f"No frame timestamps found for {animal}. Generating uniform timestamps.")
                num_frames = self.get_num_frames() * self.frame_averaging
                frame_ts = np.arange(num_frames) * (1000 / self.frame_rate)
        first_frame = np.array([0])
        last_frame = np.array([int(np.max(frame_ts) + (500 * (1000 / self.frame_rate)))])
        frame_index_temp = np.concatenate((first_frame, frame_ts, last_frame))
        frames_missed = []
        inter_frame_ms = 1000 / self.frame_rate  # ~33.333
        for i in range(len(frame_index_temp) - 1):
            num_missed = int(np.round((frame_index_temp[i + 1] - frame_index_temp[i]) / inter_frame_ms) - 1)
            if num_missed > 0:
                for j in range(num_missed):
                    missed = frame_index_temp[i] + int(inter_frame_ms * (j + 1))
                    frames_missed.append(missed)
        corrected = np.sort(np.concatenate((frame_index_temp, frames_missed)))

        return corrected[::self.frame_averaging]  # downsample

    def _get_frame_indices(self, events: NDArray[np.float64]) -> NDArray[np.float64]:
        frame_ts = self._get_frame_timestamps(self.mat_no_frames,self.animal_name)
        frame_indices = np.zeros(len(events), dtype=np.uint64)
        for i, e in enumerate(events):
            if np.isnan(e):
                frame_indices[i] = 0  # set invalid events to 0; will be filtered later
                continue
            temp = np.nonzero(frame_ts <= e)[0]
            if temp.size > 0:
                frame_indices[i] = temp[-1]
            else:
                frame_indices[i] = 0  # no frame timestamp found; use 0

        return frame_indices

    def get_animal_name(self) -> str:
        return self.animal_name

    def get_event_ids(self) -> NDArray[np.float64]:
        return self.event_ids

    def get_event_timestamps(self) -> NDArray[np.float64]:
        return self.event_timestamps

    def get_num_neurons(self) -> int:
        return self.num_neurons

    def get_num_frames(self) -> int:
        return self.num_frames

    def get_eventlog(self) -> NDArray[np.float64]:
        return self.eventlog

    def get_extracted_signals(self) -> NDArray[np.float64]:
        return self.extracted_signals

    def get_aligned_timeline(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        events = self._get_valid_events()

        if self.get_num_frames() == 0 or self.get_num_neurons() == 0:
            raise ValueError("No valid signal data available")

        # calculate frame indices from timestamps
        raw_frame_indices = np.floor(events / (1000 / self.frame_rate)).astype(np.uint64)
        frame_indices = (raw_frame_indices // self.frame_averaging).astype(np.uint64)
        frame_indices = np.clip(frame_indices, 0, self.get_num_frames() - 1)

        # convert frames to time (seconds)
        signal_time = np.arange(self.get_num_frames()) * (self.time_per_frame * self.frame_averaging)
        event_time = frame_indices * (self.time_per_frame * self.frame_averaging)

        return signal_time, event_time

    def get_event_windows(self) -> NDArray[np.float64]:
        events = self._get_valid_events()

        if self.get_num_frames() == 0 or self.get_num_neurons() == 0:
            raise ValueError("No valid signal data available")

        if events.ndim > 1:  # handle NaN array case for low events
            return events

        frame_indices = self._get_frame_indices(events)
        frame_indices = np.clip(frame_indices, 0, self.get_num_frames() - 1)

        num_trials = len(events)
        aligned_windows = np.nan * np.zeros((num_trials, self.window_size, self.get_num_neurons()), dtype=np.float64)
        signals_t = self.extracted_signals.T  # frames × neurons
        valid_trials = []
        for i, event_idx in enumerate(frame_indices):
            if np.isnan(event_idx) or event_idx < self.pre_window_size or event_idx >= (self.get_num_frames() - self.post_window_size):
                continue
            aligned_windows[i] = signals_t[int(event_idx) - self.pre_window_size:int(event_idx) + self.post_window_size]
            valid_trials.append(i)

        if valid_trials:
            aligned_windows = aligned_windows[valid_trials]
            aligned_windows = np.swapaxes(aligned_windows, 0, 2)  # neurons × window_size × trials
        else:
            aligned_windows = np.nan * np.ones((self.get_num_neurons(), self.window_size, 2), dtype=np.float64)

        return aligned_windows

    def get_pre_window_size(self) -> int:
        return self.pre_window_size

    def get_baseline_range(self) -> np.ndarray:
        return self.baseline_range

    def __str__(self):
        return f"Animal: {self.animal_name}, n={self.num_neurons}, s={len(self._get_valid_events())}"

class NSyncPopulation:
    def __init__(
        self,
        samples: List[NSyncSample],
        subtract_baseline: bool = False,
        z_scored: bool = False,
        compute_significance: bool = False,
        bh_correction: bool = False,
    ):
        self.samples = samples
        self.subtract_baseline = subtract_baseline
        self.z_scored = z_scored
        self.compute_significance = compute_significance
        self.bh_correction = bh_correction

        if not self.samples:
            self._initialize_empty()
            return

        self.baseline_range = self.samples[0].get_baseline_range()
        self.pre_window_size = self.samples[0].get_pre_window_size()

        # extract windows from each sample (filter invalid/empty)
        windows_list = []
        self.used_samples = []
        for sample in self.samples:
            if sample.get_num_events() > sample.min_events and sample.get_num_frames() > 0 and sample.get_num_neurons() > 0:
                windows_list.append(sample.get_event_windows())
                self.used_samples.append(sample)

        if not windows_list:
            self._initialize_empty()
            return

        self.max_trials = max(w.shape[2] for w in windows_list)
        self.window_size = windows_list[0].shape[1]

        self.stacked_windows = self._stack_windows(windows_list, self.max_trials)
        self.per_neuron_means = np.nanmean(self.stacked_windows, axis=2)

        # preprocessing pipeline
        if subtract_baseline:
            self.per_neuron_means = self._subtract_baseline(self.per_neuron_means, self.baseline_range)
        if self.z_scored:
            self.per_neuron_means = self._zscore_data(self.per_neuron_means, self.baseline_range)
        if self.compute_significance:
            self.significance_results = self._compute_significance(bh_correction=self.bh_correction)

        self.mean_responses = np.nanmean(self.per_neuron_means, axis=1)

    @staticmethod
    def _stack_windows(windows: List[NDArray[np.float64]], max_trials: int) -> NDArray[np.float64]:
        stacked_windows = []
        for window_set in windows:
            if window_set.shape[2] < max_trials:
                padded = np.pad(
                    window_set,
                    ((0, 0), (0, 0), (0, max_trials - window_set.shape[2])),
                    mode='constant',
                    constant_values=np.nan
                )
            else:
                padded = window_set
            stacked_windows.append(padded)
        return np.concatenate(stacked_windows, axis=0) if stacked_windows else np.array([])

    @staticmethod
    def _subtract_baseline(per_neuron_means: NDArray[np.float64], baseline_range: np.ndarray, stacked_windows: np.ndarray = None, per_trial: bool = False) -> NDArray[np.float64]:
        if per_trial and stacked_windows is not None:
            baselines = np.nanmean(stacked_windows[:, baseline_range, :], axis=1)  # neurons x trials
            return stacked_windows - baselines[:, np.newaxis, :]  # returns 3D
        else:
            baseline = np.nanmean(per_neuron_means[:, baseline_range], axis=1)[:, None]
            return per_neuron_means - baseline

    @staticmethod
    def _zscore_data(per_neuron_means: NDArray[np.float64], baseline_range: np.ndarray) -> NDArray[np.float64]:
        standardized_windows = per_neuron_means.copy()
        for i in range(standardized_windows.shape[0]):
            baseline_std = np.nanstd(standardized_windows[i, baseline_range])
            if baseline_std > 0:
                standardized_windows[i] /= baseline_std
        return standardized_windows

    def _initialize_empty(self) -> None:
        self.max_trials = 0
        self.window_size = 0
        self.stacked_windows = np.array([])
        self.per_neuron_means = np.array([])
        self.mean_responses = np.array([])

        print("No valid windows found in samples.")

    def _compute_significance(self, auc_window: tuple = (-5, 5), alpha: float = 0.05, bh_correction: bool = False) -> NDArray[np.float64]:
        sampling_rate = self.samples[0].sampling_rate
        auc_start = self.pre_window_size + int(auc_window[0] * sampling_rate)
        auc_end = self.pre_window_size + int(auc_window[1] * sampling_rate)
        baseline_range = self.baseline_range

        baselines = np.nanmean(self.stacked_windows[:, baseline_range, :], axis=1)  # neurons x trials
        stacked_sub = self.stacked_windows - baselines[:, np.newaxis, :]  # Subtract per trial

        trial_baselines = np.nanmean(stacked_sub[:, baseline_range, :], axis=1)  # neurons x trials
        trial_events = np.nanmean(stacked_sub[:, auc_start:auc_end, :], axis=1)  # neurons x trials

        num_neurons = self.stacked_windows.shape[0]
        auc_vals = np.full(num_neurons, np.nan)
        p_vals = np.full(num_neurons, np.nan)

        for n in range(num_neurons):
            x, y = trial_events[n], trial_baselines[n]
            valid_mask = ~(np.isnan(x) | np.isnan(y))
            if np.sum(valid_mask) < 2: continue  # Skip low data
            x, y = x[valid_mask], y[valid_mask]

            # Mann-Whitney U p-value
            _, p = stats.mannwhitneyu(x, y, alternative='two-sided')
            p_vals[n] = p

            # AUROC (shifted to -1 to 1)
            labels = np.concatenate([np.ones_like(x), np.zeros_like(y)])
            data = np.concatenate([x, y])
            auc = 2 * (roc_auc_score(labels, data) - 0.5)
            auc_vals[n] = auc

        if bh_correction:
            p_vals = self._benjamini_hochberg(p_vals, alpha)

        sig_mask = p_vals <= alpha

        return self.per_neuron_means[sig_mask]

    @staticmethod
    def _benjamini_hochberg(pvals, alpha=0.05) -> NDArray[np.float64]:
        finite_p = pvals[np.isfinite(pvals)]
        if finite_p.size == 0: return pvals
        sorted_idx = np.argsort(finite_p)
        sorted_p = finite_p[sorted_idx]
        m = len(sorted_p)
        thresholds = np.arange(1, m + 1) * (alpha / m)
        reject = sorted_p <= thresholds
        max_k = np.max(np.where(reject)[0]) if np.any(reject) else -1
        corrected = np.copy(pvals)
        corrected[np.argsort(pvals)[max_k + 1:]] = 1  # set non-significant to 1
        corrected[~np.isfinite(pvals)] = np.nan

        return corrected

    def valid_trials(self) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
        valid_mask = ~np.isnan(self.mean_responses)
        num_valid_neurons = 0

        if not np.any(valid_mask):
            print(f"No valid neurons found")
        else:
            self.per_neuron_means = self.per_neuron_means[valid_mask]
            self.mean_responses = self.mean_responses[valid_mask]
            num_valid_neurons = self.per_neuron_means.shape[0]

        return self.per_neuron_means, self.mean_responses, num_valid_neurons

    def sorted_indices(self) -> NDArray[np.int64]:
        sort_indices = np.argsort(self.mean_responses)[::-1]
        return sort_indices

    def __str__(self):
        animals = [a.get_animal_name() for a in self.samples]
        neurons = [a.get_num_neurons() for a in self.samples]
        events = [a.get_num_events() for a in self.samples]
        included = [True if a in self.used_samples else False for a in self.samples]
        df = pd.DataFrame.from_dict({"Animals": animals, "Neurons": neurons, "Events": events, "Included": included})

        return str(df)

if __name__ == "__main__":
    sns.set_style('white')

    root = "../data"
    output_path = os.path.join("../", "output")
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
                            mat_no_frames=os.path.join(root, 'empty.mat'),
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
                continue

            day_dataset = NSyncPopulation(
                day_samples,
                subtract_baseline=True,
                z_scored=True,
                compute_significance=True,
                bh_correction=True,
            )
            print(day)
            print(day_dataset)
            print()
            per_neuron_means, mean_responses, num_valid_neurons = day_dataset.valid_trials()
            per_neuron_means = per_neuron_means[day_dataset.sorted_indices()]
            day_datasets.append(day_dataset)
            day_num_neurons_list.append(num_valid_neurons)

        if not day_datasets:
            exit()

        num_days = len(days)
        fig, axes = plt.subplots(2, num_days, figsize=(15, 8), sharex='col', squeeze=False)
        fig.suptitle("Multi-Day Neural Activity", color='k', fontsize=16)

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
            ax1.set_title(f'{day} (n={num_valid_neurons})', color='k')
            ax1.axvline(x=day_dataset.pre_window_size, color='k', linestyle='--', alpha=0.7, label='Event')
            ax1.tick_params(colors='k')

            ax2 = axes[1, idx]
            ax2.set_ylim(-.5, 2.5)
            ax2.plot(np.nanmean(per_neuron_means, axis=0), color='r', label='Mean Activity')
            ax2.axvline(x=day_dataset.pre_window_size, color='k', linestyle='--', alpha=0.7, label='Event')
            ax2.set_xlabel('Frames Relative to Event', color='k')
            ax2.grid(True, color='gray', alpha=0.3)
            ax2.tick_params(colors='k')

            if idx == 0:
                ax1.set_ylabel('Neurons', color='k')
                ax2.set_ylabel('Mean Response', color='k')

        plt.tight_layout()
        plt.show()