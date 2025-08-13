import warnings
from typing import Any, Dict, List, Union
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import scipy.io as sio

class NSyncDataset:
    def __init__(
        self,
        eventlog: List[Union[str, Path]] | Dict[str, Any],
        extracted_signals: List[Union[str, Path]] | NDArray,
        animal_name: str = "REX",
    ) -> None:
        self.eventlog = eventlog
        self.extracted_signals = extracted_signals
        self.animal_name = animal_name

        if isinstance(eventlog, list):
            self.eventlog = sorted([str(f) for f in eventlog])
            self.eventlog = self._compile_matlab_files(eventlog)
        else:
            self.eventlog = eventlog
        self.event_ids = self.eventlog[:, 0]
        self.event_timestamps = self.eventlog[:, 1]

        if isinstance(extracted_signals, list):
            self.extracted_signals = sorted([str(f) for f in extracted_signals])
            self.extracted_signals = self._compile_npy_files(extracted_signals)
        else:
            self.extracted_signals = extracted_signals.astype(np.float64)

    @staticmethod
    def _compile_matlab_files(files: list) -> NDArray[np.float64]:
        stack = []
        last_timestamp = 0
        if isinstance(files, list) and len(files) > 0:
            for file in files:
                try:
                    data = sio.loadmat(file)
                    eventlog = np.squeeze(data["eventlog"])
                    eventlog[:, 1] = eventlog[:, 1] + last_timestamp # offset timestamps
                    last_timestamp = np.max(eventlog[:, 1])

                    stack.append(eventlog[:, 0:2])
                except Exception as e:
                    print(f"Error loading {file}: {e}")
            stack = np.vstack(stack).squeeze() if len(stack) > 0 else np.squeeze(stack)
            stack = stack[stack[:, 0] != 0] # only return valid data

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

    def event_ids(self):
        return self.event_ids

    def event_timestamps(self):
        return self.event_timestamps

    def num_neurons(self) -> int:
        return self.extracted_signals.shape[0]

    def num_frames(self) -> int:
        return self.extracted_signals.shape[1]

    def eventlog(self):
        return self.eventlog

    def extracted_signals(self):
        return self.extracted_signals

class NSyncTimeline(NSyncDataset):
    def __init__(
        self,
        eventlog: List[Union[str, Path]] | Dict[str, Any],
        extracted_signals: List[Union[str, Path]] | NDArray[np.uint64],
        target_event_id: Union[int, List[int]] = 22,
        noise_event_id: Union[int, List[int]] = 222,
        frame_rate: int = 30,
        frame_averaging: int = 4,
        isolated_events: bool = False,
        min_events: int = 3,
        normalize_data: bool = True
    ):
        super().__init__(eventlog, extracted_signals)

        # configure data acquisition variables
        self.frame_rate = frame_rate
        self.frame_averaging = frame_averaging
        self.target_event_id = target_event_id
        self.noise_event_id = noise_event_id
        self.min_events = min_events
        self.isolated_events = isolated_events
        self.sampling_rate = frame_rate / frame_averaging
        self.time_per_frame = 1 / frame_rate

        # configure window of interest
        self.pre_window_size = int(10 * self.sampling_rate)
        self.window_size = int((self.pre_window_size * 2) + (1.6 * self.sampling_rate))
        self.post_window_size = self.window_size - self.pre_window_size

        # configure baseline and normalization
        self.baseline_range = np.arange(int(3 * self.sampling_rate))
        self.normalize_data = normalize_data

        # identify event IDs
        self.eventlog_dict = {
            "active_lever": 22,
            "active_lever_timeout": 222,
            "inactive_lever": 21,
            "inactive_lever_timeout": 212,
            "cue": 7,
            "infusion": 4,
        }

        # normalize signals (dF/F)
        if self.normalize_data:
            self.extracted_signals = self._normalize_signals(self.extracted_signals)

    @staticmethod
    def _normalize_signals(extracted_signals: NDArray[np.uint64]):
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

    def _get_valid_events(self, sep: float = 1000.0) -> np.ndarray:
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

    def _get_frame_timestamps(self, animal: str = None) -> np.ndarray:
        frame_ts = self.event_timestamps[self.event_ids == 9]
        if frame_ts.size == 0:
            if animal in ['CTL1', 'ER-L1', 'ER-L2', 'IG-19', 'IG-28', 'PGa-T1', 'XYZ']:
                print(f"No frame timestamps found for {animal}. Attempting to use assumed timestamps.")
                try:
                    assumed_data = sio.loadmat('path/to/matfile_noframes_3.mat')  # Update with actual path
                    frame_ts = np.squeeze(assumed_data['eventlog'])[np.squeeze(assumed_data['eventlog'])[:, 0] == 9, 1]
                except FileNotFoundError:
                    print(f"Assumed timestamps file not found. Generating uniform timestamps based on signal length.")
                    num_frames = self.num_frames() * self.frame_averaging
                    frame_ts = np.arange(num_frames) * (1000 / self.frame_rate)  # Timestamps in ms
            else:
                print(f"No frame timestamps found for {animal}. Generating uniform timestamps.")
                num_frames = self.num_frames() * self.frame_averaging
                frame_ts = np.arange(num_frames) * (1000 / self.frame_rate)

        first_frame = np.array([0])
        last_frame = np.array([int(np.max(frame_ts) + (500 * (1000 / self.frame_rate)))])
        frame_index_temp = np.concatenate((first_frame, frame_ts, last_frame))
        frames_missed = []
        inter_frame_ms = 1000 / self.frame_rate
        for i in range(len(frame_index_temp) - 1):
            num_missed = int(np.round((frame_index_temp[i + 1] - frame_index_temp[i]) / inter_frame_ms) - 1)
            if num_missed > 0:
                for j in range(num_missed):
                    missed = frame_index_temp[i] + int(inter_frame_ms * (j + 1))
                    frames_missed.append(missed)
        corrected = np.sort(np.concatenate((frame_index_temp, frames_missed)))
        return corrected[::self.frame_averaging]

    def _get_frame_indices(self, events: np.ndarray) -> np.ndarray:
        frame_ts = self._get_frame_timestamps(self.animal_name)
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

    def aligned_timeline(self) -> tuple[np.ndarray, np.ndarray]:
        events = self._get_valid_events()

        if self.num_frames() == 0 or self.num_neurons() == 0:
            raise ValueError("No valid signal data available")

        # calculate frame indices from timestamps
        raw_frame_indices = np.floor(events / (1000 / self.frame_rate)).astype(np.uint64)
        frame_indices = (raw_frame_indices // self.frame_averaging).astype(np.uint64)
        frame_indices = np.clip(frame_indices, 0, self.num_frames() - 1)

        # convert frames to time (seconds)
        signal_time = np.arange(self.num_frames()) * (self.time_per_frame * self.frame_averaging)
        event_time = frame_indices * (self.time_per_frame * self.frame_averaging)

        return signal_time, event_time

    def event_windows(self) -> NDArray[np.float64]:
        events = self._get_valid_events()

        if self.num_frames() == 0 or self.num_neurons() == 0:
            raise ValueError("No valid signal data available")

        if events.ndim > 1:  # handle NaN array case for low events
            return events

        frame_indices = self._get_frame_indices(events)
        frame_indices = np.clip(frame_indices, 0, self.num_frames() - 1)

        num_trials = len(events)
        aligned_windows = np.nan * np.zeros((num_trials, self.window_size, self.num_neurons()), dtype=np.float64)
        signals_t = self.extracted_signals.T  # frames × neurons
        valid_trials = []
        for i, event_idx in enumerate(frame_indices):
            if np.isnan(event_idx) or event_idx < self.pre_window_size or event_idx >= (self.num_frames() - self.post_window_size):
                continue
            aligned_windows[i] = signals_t[int(event_idx) - self.pre_window_size:int(event_idx) + self.post_window_size]
            valid_trials.append(i)

        if valid_trials:
            aligned_windows = aligned_windows[valid_trials]
            aligned_windows = np.swapaxes(aligned_windows, 0, 2)  # neurons × window_size × trials
        else:
            aligned_windows = np.nan * np.ones((self.num_neurons(), self.window_size, 2), dtype=np.float64)

        return aligned_windows