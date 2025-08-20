# from scipy import stats
# from sklearn.metrics import roc_auc_score
# import pandas as pd
# import warnings
# warnings.filterwarnings('always', category=UserWarning)
# warnings.filterwarnings('always', category=DeprecationWarning)
#
# import numpy as np
# from numpy.typing import NDArray
# from typing import List
# from nsync2p.sample import NSyncSample
#
# class NSyncPopulation:
#     def __init__(
#         self,
#         samples: List[NSyncSample],
#         subtract_baseline: bool = False,
#         z_scored: bool = False,
#         compute_significance: bool = False,
#         bh_correction: bool = False,
#     ):
#         self._samples = samples
#         self.subtract_baseline = subtract_baseline
#         self.z_scored = z_scored
#         self.compute_significance = compute_significance
#         self.bh_correction = bh_correction
#
#         if not self._samples:
#             self._initialize_empty()
#             return
#
#         self.baseline_range = self._samples[0].get_baseline_range()
#         self.pre_window_size = self._samples[0].get_pre_window_size()
#
#         # extract windows from each sample (filter invalid/empty)
#         windows_list = []
#         self.used_samples = []
#         for sample in self._samples:
#             if sample.get_num_events() > sample.min_events and sample.get_num_frames() > 0 and sample.get_num_neurons() > 0:
#                 windows_list.append(sample.get_event_windows())
#                 self.used_samples.append(sample)
#
#         if not windows_list:
#             self._initialize_empty()
#             return
#
#         self.max_trials = max(w.shape[2] for w in windows_list)
#         self.window_size = windows_list[0].shape[1]
#
#         self.stacked_windows = self._stack_windows(windows_list, self.max_trials)
#         self.per_neuron_means = np.nanmean(self.stacked_windows, axis=2)
#
#         # preprocessing pipeline
#         if subtract_baseline:
#             self.per_neuron_means = self._subtract_baseline(self.per_neuron_means, self.baseline_range)
#         if self.z_scored:
#             self.per_neuron_means = self._zscore_data(self.per_neuron_means, self.baseline_range)
#         if self.compute_significance:
#             self.significance_results = self._compute_significance(bh_correction=self.bh_correction)
#
#         self.mean_responses = np.nanmean(self.per_neuron_means, axis=1)
#
#     @staticmethod
#     def _stack_windows(windows: List[NDArray[np.float64]], max_trials: int) -> NDArray[np.float64]:
#         stacked_windows = []
#         for window_set in windows:
#             if window_set.shape[2] < max_trials:
#                 padded = np.pad(
#                     window_set,
#                     ((0, 0), (0, 0), (0, max_trials - window_set.shape[2])),
#                     mode='constant',
#                     constant_values=np.nan
#                 )
#             else:
#                 padded = window_set
#             stacked_windows.append(padded)
#         return np.concatenate(stacked_windows, axis=0) if stacked_windows else np.array([])
#
#     @staticmethod
#     def _subtract_baseline(per_neuron_means: NDArray[np.float64], baseline_range: np.ndarray, stacked_windows: np.ndarray = None, per_trial: bool = False) -> NDArray[np.float64]:
#         if per_trial and stacked_windows is not None:
#             baselines = np.nanmean(stacked_windows[:, baseline_range, :], axis=1)  # neurons x trials
#             return stacked_windows - baselines[:, np.newaxis, :]  # returns 3D
#         else:
#             baseline = np.nanmean(per_neuron_means[:, baseline_range], axis=1)[:, None]
#             return per_neuron_means - baseline
#
#     @staticmethod
#     def _zscore_data(per_neuron_means: NDArray[np.float64], baseline_range: np.ndarray) -> NDArray[np.float64]:
#         standardized_windows = per_neuron_means.copy()
#         for i in range(standardized_windows.shape[0]):
#             baseline_std = np.nanstd(standardized_windows[i, baseline_range])
#             if baseline_std > 0:
#                 standardized_windows[i] /= baseline_std
#         return standardized_windows
#
#     def _initialize_empty(self) -> None:
#         self.max_trials = 0
#         self.window_size = 0
#         self.stacked_windows = np.array([])
#         self.per_neuron_means = np.array([])
#         self.mean_responses = np.array([])
#
#         print("No valid windows found in samples.")
#
#     def _compute_significance(self, auc_window: tuple = (-5, 5), alpha: float = 0.05, bh_correction: bool = False) -> NDArray[np.float64]:
#         sampling_rate = self._samples[0].sampling_rate
#         auc_start = self.pre_window_size + int(auc_window[0] * sampling_rate)
#         auc_end = self.pre_window_size + int(auc_window[1] * sampling_rate)
#         baseline_range = self.baseline_range
#
#         if self.stacked_windows[:, baseline_range, :].size > 0:
#             baselines = np.nanmean(self.stacked_windows[:, baseline_range, :], axis=1)  # neurons x trials
#             stacked_sub = self.stacked_windows - baselines[:, np.newaxis, :]  # Subtract per trial
#
#             trial_baselines = np.nanmean(stacked_sub[:, baseline_range, :], axis=1)  # neurons x trials
#             trial_events = np.nanmean(stacked_sub[:, auc_start:auc_end, :], axis=1)  # neurons x trials
#         else:
#             trial_baselines = np.nan * np.ones((self.stacked_windows.shape[0], self.max_trials))
#             trial_events = np.nan * np.ones((self.stacked_windows.shape[0], self.max_trials))
#
#         num_neurons = self.stacked_windows.shape[0]
#         auc_vals = np.full(num_neurons, np.nan)
#         p_vals = np.full(num_neurons, np.nan)
#
#         for n in range(num_neurons):
#             x, y = trial_events[n], trial_baselines[n]
#             valid_mask = ~(np.isnan(x) | np.isnan(y))
#             if np.sum(valid_mask) < 2: continue  # Skip low data
#             x, y = x[valid_mask], y[valid_mask]
#
#             # Mann-Whitney U p-value
#             _, p = stats.mannwhitneyu(x, y, alternative='two-sided')
#             p_vals[n] = p
#
#             # AUROC (shifted to -1 to 1)
#             labels = np.concatenate([np.ones_like(x), np.zeros_like(y)])
#             data = np.concatenate([x, y])
#             auc = 2 * (roc_auc_score(labels, data) - 0.5)
#             auc_vals[n] = auc
#
#         if bh_correction:
#             p_vals = self._benjamini_hochberg(p_vals, alpha)
#
#         sig_mask = p_vals <= alpha
#
#         return self.per_neuron_means[sig_mask]
#
#     @staticmethod
#     def _benjamini_hochberg(pvals, alpha=0.05) -> NDArray[np.float64]:
#         finite_p = pvals[np.isfinite(pvals)]
#         if finite_p.size == 0: return pvals
#         sorted_idx = np.argsort(finite_p)
#         sorted_p = finite_p[sorted_idx]
#         m = len(sorted_p)
#         thresholds = np.arange(1, m + 1) * (alpha / m)
#         reject = sorted_p <= thresholds
#         max_k = np.max(np.where(reject)[0]) if np.any(reject) else -1
#         corrected = np.copy(pvals)
#         corrected[np.argsort(pvals)[max_k + 1:]] = 1  # set non-significant to 1
#         corrected[~np.isfinite(pvals)] = np.nan
#
#         return corrected
#
#     def valid_trials(self) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
#         valid_mask = ~np.isnan(self.mean_responses)
#         num_valid_neurons = 0
#
#         if not np.any(valid_mask):
#             print(f"No valid neurons found")
#         else:
#             self.per_neuron_means = self.per_neuron_means[valid_mask]
#             self.mean_responses = self.mean_responses[valid_mask]
#             num_valid_neurons = self.per_neuron_means.shape[0]
#
#         return self.per_neuron_means, self.mean_responses, num_valid_neurons
#
#     def sorted_indices(self) -> NDArray[np.int64]:
#         sort_indices = np.argsort(self.mean_responses)[::-1]
#         return sort_indices
#
#     def __str__(self):
#         animals = [a.get_animal_name() for a in self._samples]
#         neurons = [a.get_num_neurons() for a in self._samples]
#         events = [a.get_num_events() for a in self._samples]
#         included = [True if a in self.used_samples else False for a in self._samples]
#         df = pd.DataFrame.from_dict({"Animals": animals, "Neurons": neurons, "Events": events, "Included": included})
#
#         return str(df)

from scipy import stats
from sklearn.metrics import roc_auc_score
import pandas as pd
import warnings
warnings.filterwarnings('always', category=UserWarning)
warnings.filterwarnings('always', category=DeprecationWarning)

import numpy as np
from numpy.typing import NDArray
from typing import List
from nsync2p.sample import NSyncSample

class NSyncPopulation:
    def __init__(
        self,
        samples: List[NSyncSample],
        normalize_signals: bool = False,
        subtract_baseline: bool = False,
        z_scored: bool = False,
        compute_significance: bool = False,
        bh_correction: bool = False,
    ):
        self.samples = samples
        self.normalize_signals = normalize_signals
        self.subtract_baseline = subtract_baseline
        self.z_scored = z_scored
        self.compute_significance = compute_significance
        self.bh_correction = bh_correction

        if not self.samples:
            self._initialize_empty()
            return

        # extract windows from each sample (filter invalid/empty)
        windows_list = []
        self.used_samples = []
        self.max_trials = 0
        for sample in self.samples:
            if sample.get_num_events() > sample.min_events and sample.get_num_frames() > 0 and sample.get_num_neurons() > 0:
                windows_list.append(sample.get_event_windows())
                num_trials = sample.get_num_events()
                if num_trials > self.max_trials:
                    self.max_trials = num_trials
                self.used_samples.append(sample)

        if not windows_list:
            self._initialize_empty()
            return

        # assume get_window_dimensions exists or replace with appropriate attributes
        self.pre_window_size, self.post_window_size, self.window_size = self.samples[0].get_window_dimensions()
        self.sampling_rate = self.samples[0].get_sampling_rate()
        self.baseline_range = np.arange(int(3 * self.sampling_rate))

        # compute raw per_neuron_means and mean_responses without stacking (low memory)
        raw_per_neuron_means_list = [np.nanmean(w, axis=2) for w in windows_list]
        self.raw_per_neuron_means = np.concatenate(raw_per_neuron_means_list, axis=0) if raw_per_neuron_means_list else np.array([])
        self.raw_mean_responses = np.nanmean(self.raw_per_neuron_means, axis=1)

        # apply the preprocessing pipeline based on init params
        self._pipeline()

    def _pipeline(self) -> None:
        """Apply the preprocessing pipeline based on class parameters, updating processed attributes while preserving raw data."""
        # get appropriate windows and compute per_neuron_means (low memory: avoid full stack unless needed for significance)
        windows_list = [sample.get_event_windows() for sample in self.used_samples]

        if self.normalize_signals:
            windows_list = [self._normalize_windows(w) for w in windows_list]

        per_neuron_means_list = [np.nanmean(w, axis=2) for w in windows_list]
        self.per_neuron_means = np.concatenate(per_neuron_means_list, axis=0) if per_neuron_means_list else np.array([])

        stacked = None
        if self.compute_significance:
            stacked = self._stack_windows(windows_list, self.max_trials)

        if self.subtract_baseline:
            self.per_neuron_means = self._subtract_baseline(self.per_neuron_means, self.baseline_range)

        if self.z_scored:
            self.per_neuron_means = self._zscore_data(self.per_neuron_means, self.baseline_range)

        self.mean_responses = np.nanmean(self.per_neuron_means, axis=1)

        self.significance_results = None
        if self.compute_significance:
            sig_mask = self._compute_significance(stacked=stacked)
            self.significance_results = self.per_neuron_means[sig_mask]

    @staticmethod
    def _normalize_windows(windows: NDArray[np.float64]) -> NDArray[np.float64]:
        means = np.nanmean(windows, axis=(1, 2), keepdims=True)
        normalized = np.divide(
            windows,
            means,
            out=np.zeros_like(windows),
            where=(means != 0) & (~np.isnan(means))
        )
        means = np.nanmean(normalized, axis=(1, 2), keepdims=True)
        stds = np.nanstd(normalized, axis=(1, 2), keepdims=True)
        normalized = np.divide(
            normalized - means,
            stds,
            out=np.zeros_like(normalized),
            where=stds > 0
        )
        return normalized

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
    def _subtract_baseline(per_neuron_means: NDArray[np.float64], baseline_range: np.ndarray) -> NDArray[np.float64]:
        baseline = np.nanmean(per_neuron_means[:, baseline_range], axis=1, keepdims=True)
        return per_neuron_means - baseline

    @staticmethod
    def _zscore_data(per_neuron_means: NDArray[np.float64], baseline_range: np.ndarray) -> NDArray[np.float64]:
        baseline_stds = np.nanstd(per_neuron_means[:, baseline_range], axis=1, keepdims=True)
        return np.divide(
            per_neuron_means,
            baseline_stds,
            out=np.zeros_like(per_neuron_means),
            where=baseline_stds > 0
        )

    def _initialize_empty(self) -> None:
        self.max_trials = 0
        self.window_size = 0
        self.raw_per_neuron_means = np.array([])
        self.raw_mean_responses = np.array([])
        self.per_neuron_means = np.array([])
        self.mean_responses = np.array([])

        print("No valid windows found in samples.")

    def _compute_significance(self, stacked: NDArray[np.float64], auc_window: tuple = (-5, 5), alpha: float = 0.05) -> NDArray[bool]:
        auc_start = self.pre_window_size + int(auc_window[0] * self.sampling_rate)
        auc_end = self.pre_window_size + int(auc_window[1] * self.sampling_rate)

        if stacked[:, self.baseline_range, :].size == 0:
            baselines = np.nanmean(stacked[:, self.baseline_range, :], axis=1)  # neurons x trials
            stacked_sub = stacked - baselines[:, np.newaxis, :]  # Subtract per trial

            trial_baselines = np.nanmean(stacked_sub[:, self.baseline_range, :], axis=1)  # neurons x trials
            trial_events = np.nanmean(stacked_sub[:, auc_start:auc_end, :], axis=1)  # neurons x trials
        else:
            trial_baselines = np.full((stacked.shape[0], self.max_trials), np.nan)
            trial_events = np.full((stacked.shape[0], self.max_trials), np.nan)

        num_neurons = stacked.shape[0]
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

        if self.bh_correction:
            p_vals = self._benjamini_hochberg(p_vals, alpha)

        sig_mask = p_vals <= alpha

        return sig_mask

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

    def get_valid_trials(self) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
        valid_mask = ~np.isnan(self.mean_responses)
        num_valid_neurons = 0

        if not np.any(valid_mask):
            print(f"No valid neurons found")
        else:
            self.per_neuron_means = self.per_neuron_means[valid_mask]
            self.mean_responses = self.mean_responses[valid_mask]
            num_valid_neurons = self.per_neuron_means.shape[0]

        return self.per_neuron_means, self.mean_responses, num_valid_neurons

    def get_sorted_indices(self) -> NDArray[np.int64]:
        sort_indices = np.argsort(self.mean_responses)[::-1]
        return sort_indices

    def __str__(self):
        animals = [a.get_animal_name() for a in self.samples]
        neurons = [a.get_num_neurons() for a in self.samples]
        events = [a.get_num_events() for a in self.samples]
        included = [True if a in self.used_samples else False for a in self.samples]
        df = pd.DataFrame.from_dict({"Animals": animals, "Neurons": neurons, "Events": events, "Included": included})

        return str(df)