from scipy import stats
from sklearn.metrics import roc_auc_score
import pandas as pd
from tqdm import tqdm
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
        name: str = "POP",
        subtract_baseline: bool = False,
        z_score_baseline: bool = False,
        compute_significance: bool = False,
        bh_correction: bool = False,
    ):
        self._name = name
        self._samples = samples
        self._subtract_baseline = subtract_baseline
        self._z_score_baseline = z_score_baseline
        self._compute_significance = compute_significance
        self._bh_correction = bh_correction

        if not self._samples:
            self.__initialize_empty__()
            return

        # extract windows from each sample (filter invalid/empty)
        windows_list = []
        self._used_samples = []
        self._max_trials = 0
        for sample in self._samples:
            if sample.get_num_events() > sample.get_min_events() and sample.get_num_frames() > 0 and sample.get_num_neurons() > 0:
                windows_list.append(sample.get_event_windows())
                num_trials = sample.get_num_events()
                if num_trials > self._max_trials:
                    self._max_trials = num_trials
                self._used_samples.append(sample)

        if not windows_list:
            self.__initialize_empty__()
            return

        # assume get_window_dimensions exists or replace with appropriate attributes
        self._pre_window_size, self._post_window_size, self._window_size = self._samples[0].get_window_dimensions()
        self._sampling_rate = self._samples[0].get_sampling_rate()
        self._baseline_range = np.arange(int(3 * self._sampling_rate))

        # compute raw per_neuron_means and mean_responses without stacking (low memory)
        raw_per_neuron_means_list = [np.nanmean(w, axis=2) for w in windows_list]
        self._raw_per_neuron_means = np.concatenate(raw_per_neuron_means_list, axis=0) if raw_per_neuron_means_list else np.array([])
        self._raw_mean_responses = np.nanmean(self._raw_per_neuron_means, axis=1)

        # apply the preprocessing pipeline based on init params
        self.__pipeline__()


    # private
    def __pipeline__(self) -> None:
        windows_list = [sample.get_event_windows() for sample in self._used_samples]

        per_neuron_means_list = [np.nanmean(w, axis=2) for w in windows_list]
        self.per_neuron_means = np.concatenate(per_neuron_means_list, axis=0) if per_neuron_means_list else np.array([])
        stacked = self.__stack_windows__(windows_list, self._max_trials)

        if self._subtract_baseline:
            self.per_neuron_means = self.__subtract_baseline__(self.per_neuron_means, self._baseline_range)

        if self._z_score_baseline:
            self.per_neuron_means = self.__z_score_baseline__(self.per_neuron_means, self._baseline_range)

        self.mean_responses = np.nanmean(self.per_neuron_means, axis=1)

        self.significance_results = None
        if self._compute_significance:
            sig_mask = self.__compute_significance__(stacked=stacked)
            self.significance_results = self.per_neuron_means[sig_mask]

    def __stack_windows__(self, windows: List[NDArray[np.float64]], max_trials: int) -> NDArray[np.float64]:
        stacked_windows = []
        for _, window_set in enumerate(tqdm(windows, desc=f"{self._name} | Stacking windows, n={len(windows)}", total=len(windows))):
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
    def __subtract_baseline__(per_neuron_means: NDArray[np.float64], baseline_range: np.ndarray) -> NDArray[np.float64]:
        baseline = np.nanmean(per_neuron_means[:, baseline_range], axis=1, keepdims=True)
        return per_neuron_means - baseline

    @staticmethod
    def __z_score_baseline__(per_neuron_means: NDArray[np.float64], baseline_range: np.ndarray) -> NDArray[np.float64]:
        baseline_stds = np.nanstd(per_neuron_means[:, baseline_range], axis=1, keepdims=True)
        return np.divide(
            per_neuron_means,
            baseline_stds,
            out=np.zeros_like(per_neuron_means),
            where=baseline_stds > 0
        )

    def __initialize_empty__(self) -> None:
        self._max_trials = 0
        self._window_size = 0
        self._raw_per_neuron_means = np.array([])
        self._raw_mean_responses = np.array([])
        self.per_neuron_means = np.array([])
        self.mean_responses = np.array([])

        print("No valid windows found in samples.")

    def __compute_significance__(self, stacked: NDArray[np.float64], auc_window: tuple = (-5, 5),
                                 alpha: float = 0.05) -> NDArray[bool]:
        auc_start = self._pre_window_size + int(auc_window[0] * self._sampling_rate)  # -5s relative to event
        auc_end = self._pre_window_size + int(auc_window[1] * self._sampling_rate)  # +5s relative to event

        trial_baselines = np.nanmean(stacked[:, self._baseline_range, :], axis=1)  # neurons x trials
        trial_events = np.nanmean(stacked[:, auc_start:auc_end, :], axis=1)  # neurons x trials

        num_neurons = stacked.shape[0]
        auc_vals = np.full(num_neurons, np.nan)
        p_vals = np.full(num_neurons, np.nan)

        for n in tqdm(range(num_neurons), desc=f"{self._name} | Computing significance, n={num_neurons}",
                      total=num_neurons):
            x = trial_events[n]  # response data for neuron n
            y = trial_baselines[n]  # baseline data for neuron n

            # remove NaNs
            valid_mask = ~(np.isnan(x) | np.isnan(y))
            if np.sum(valid_mask) < 2:  # need at least 2 valid points for Mann-Whitney U
                continue
            x_clean, y_clean = x[valid_mask], y[valid_mask]

            # Mann-Whitney U p-value
            try:
                _, p = stats.mannwhitneyu(x_clean, y_clean, alternative='two-sided')
                p_vals[n] = p
            except ValueError:
                p_vals[n] = np.nan
                continue

            # AUROC (shifted to -1 to 1)
            try:
                labels = np.concatenate([np.ones_like(x_clean), np.zeros_like(y_clean)])
                data = np.concatenate([x_clean, y_clean])
                auc = roc_auc_score(labels, data)
                auc_vals[n] = 2 * (auc - 0.5)  # Shift to [-1, 1]
            except ValueError:
                auc_vals[n] = np.nan

        if self._bh_correction:
            p_vals = self.__benjamini_hochberg__(p_vals, alpha)

        sig_mask = p_vals <= alpha
        sig_mask[~np.isfinite(p_vals)] = False  # ensure NaN p-values are not significant

        return sig_mask

    @staticmethod
    def __benjamini_hochberg__(pvals, alpha=0.05) -> NDArray[np.float64]:
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

    # public
    def get_samples(self) -> List[NSyncSample]:
        return self._samples

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

    def get_pre_window_size(self) -> int:
        return self._pre_window_size

    def get_post_window_size(self) -> int:
        return self._post_window_size

    def get_window_size(self) -> int:
        return self._window_size

    def get_sampling_rate(self) -> float:
        return self._sampling_rate

    def get_baseline_range(self) -> np.ndarray:
        return self._baseline_range

    def __str__(self):
        animals = [a.get_animal_name() for a in self._used_samples]
        neurons = [a.get_num_neurons() for a in self._used_samples]
        events = [a.get_num_events() for a in self._used_samples]
        df = pd.DataFrame.from_dict({"Animals": animals, "Neurons": neurons, "Trials": events})

        summary = f"""
\n====================\n
Population: {self._name}
Animals: {len(self._used_samples)} / {len(self._samples)}
Neurons: {sum(neurons)}
Trials: {sum(events)}
Criteria: t >= {self._used_samples[0].get_min_events()}\n
{str(df)}
\n====================\n"""

        return summary
