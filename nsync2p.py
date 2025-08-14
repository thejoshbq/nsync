import warnings
from typing import Any, Dict, List, Union
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import scipy.io as sio

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

        # Move shape computation early for use in _get_frame_timestamps
        self.num_neurons = self.extracted_signals.shape[0]
        self.num_frames = self.extracted_signals.shape[1]

        # Sample processing logic
        self.frame_rate = frame_rate
        self.frame_averaging = frame_averaging
        self.target_event_id = target_event_id
        self.noise_event_id = noise_event_id
        self.min_events = min_events
        self.isolated_events = isolated_events
        self.sampling_rate = frame_rate / frame_averaging
        self.time_per_frame = 1 / frame_rate

        # Configure window of interest
        self.pre_window_size = int(10 * self.sampling_rate)
        self.window_size = int((self.pre_window_size * 2) + (1.6 * self.sampling_rate))
        self.post_window_size = self.window_size - self.pre_window_size

        # Configure baseline and normalization
        self.baseline_range = np.arange(int(3 * self.sampling_rate))
        self.normalize_data = normalize_data
        if self.normalize_data:  # (dF/F)
            self.extracted_signals = self._normalize_signals(self.extracted_signals)

        # Identify event IDs
        self.eventlog_dict = {
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

    def _get_frame_timestamps(self, mat_file: Union[str, Path] | None = None, animal: str = None) -> NDArray[
        np.float64]:
        frame_ts = self.event_timestamps[self.event_ids == 9]
        if frame_ts.size == 0:
            if animal in ['CTL1', 'ER-L1', 'ER-L2', 'IG-19', 'IG-28', 'PGa-T1', 'XYZ'] and mat_file is not None:
                print(f"No frame timestamps found for {animal}. Attempting to use assumed timestamps.")

                try:
                    assumed_data = sio.loadmat(mat_file)
                    eventlog_noframes = np.squeeze(assumed_data['eventlog'])

                    # Triplication to extend timestamps
                    max_of = np.max(eventlog_noframes[:, 1])
                    length_of = len(eventlog_noframes[:, 1])
                    x = np.vstack((eventlog_noframes, eventlog_noframes, eventlog_noframes))
                    x[length_of:, 1] += max_of
                    x[2 * length_of:, 1] += 2 * max_of
                    eventlog_noframes = x

                    frame_ts = eventlog_noframes[eventlog_noframes[:, 0] == 9, 1]

                    dropped_frames = []
                    diff_frames = np.diff(frame_ts)
                    inter_frame_interval = 33  # Original uses 33 (not 33.333)
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

class NSyncPopulation:
    def __init__(
        self,
        samples: List[NSyncSample],
        subtract_baseline: bool = True,
        z_scored: bool = True,
    ):
        self.samples = samples
        self.subtract_baseline = subtract_baseline
        self.z_scored = z_scored

        if not self.samples:
            self.max_trials = 0
            self.window_size = 0
            self.stacked_windows = np.array([])
            self.per_neuron_means = np.array([])
            self.mean_responses = np.array([])
            self.baseline_range = np.array([])
            self.pre_window_size = 0
            print("No samples provided; population is empty.")
            return

        # Share configs from the first sample (assume uniform across samples)
        self.baseline_range = self.samples[0].get_baseline_range()
        self.pre_window_size = self.samples[0].get_pre_window_size()

        # Extract windows from each sample (filter invalid/empty)
        windows_list = [
            sample.get_event_windows()
            for sample in self.samples
            if sample.get_event_windows().ndim == 3 and sample.get_event_windows().size > 0
        ]

        if not windows_list:
            self.max_trials = 0
            self.window_size = 0
            self.stacked_windows = np.array([])
            self.per_neuron_means = np.array([])
            self.mean_responses = np.array([])
            print("No valid windows found in samples.")
            return

        self.max_trials = max(w.shape[2] for w in windows_list)
        self.window_size = windows_list[0].shape[1]

        # Make population stack
        self.stacked_windows = self._stack_windows(windows_list, self.max_trials)

        # Calculate means per neuron in stack
        self.per_neuron_means = np.nanmean(self.stacked_windows, axis=2)

        # Calculate baseline-subtracted means
        if subtract_baseline:
            self.per_neuron_means = self._subtract_baseline(self.per_neuron_means, self.baseline_range)

        # Z-score data
        if self.z_scored:
            self.per_neuron_means = self._zscore_data(self.per_neuron_means, self.baseline_range)

        # Calculate mean across windows
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
    def _subtract_baseline(per_neuron_means: NDArray[np.float64], baseline_range: np.ndarray) -> NDArray[np.float64]:
        baseline = np.nanmean(per_neuron_means[:, baseline_range], axis=1)[:, None]
        baseline_subtracted = per_neuron_means - baseline
        return baseline_subtracted

    @staticmethod
    def _zscore_data(per_neuron_means: NDArray[np.float64], baseline_range: np.ndarray) -> NDArray[np.float64]:
        standardized_windows = per_neuron_means.copy()
        for i in range(standardized_windows.shape[0]):
            baseline_std = np.nanstd(standardized_windows[i, baseline_range])
            if baseline_std > 0:
                standardized_windows[i] /= baseline_std
            # else: remains unchanged (avoids div-by-zero, but data might be flat/NaN)
        return standardized_windows

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


