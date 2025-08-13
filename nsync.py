import warnings
from typing import Any

import numpy as np
import scipy.io as sio
from numpy import ndarray, dtype

def stack_mat(mat_list: list) -> list[Any] | ndarray[tuple[Any, ...], Any]:
    stacked_mat_files = []
    eventlog = None
    if isinstance(mat_list, list) and len(mat_list) > 0:
        for mat_file in mat_list:
            try:
                mat = np.squeeze(sio.loadmat(mat_file)["eventlog"])
                stacked_mat_files.append(mat[:, 0:2])
            except Exception as e:
                print(f"Error loading {mat_file}: {e}")
        stacked_mat_files = np.vstack(stacked_mat_files).squeeze() if len(stacked_mat_files) > 0 else np.squeeze(stacked_mat_files)
        eventlog = stacked_mat_files[stacked_mat_files[:, 0] != 0]

    return eventlog if eventlog is not None else stacked_mat_files

def stack_npy(npy_list: list) -> np.ndarray:
    signals = []
    if isinstance(npy_list, list) and len(npy_list) > 0:
        for npy_file in npy_list:
            with warnings.catch_warnings(record=True) as captured_warnings:
                warnings.simplefilter("always")

                try:
                    loaded_npy = np.load(npy_file).squeeze()
                    signals.append(loaded_npy)
                except Exception as e:
                    print(f"Error loading {npy_file}: {e}")
                    continue

                for warning in captured_warnings:
                    if "Python 2" in warning.message:
                        np.save(npy_file, loaded_npy)

        stacked_npy = np.hstack(signals)
    else:
        stacked_npy = np.array(npy_list)

    return stacked_npy if stacked_npy.ndim == 2 else None

def stack_windows(windows: np.ndarray) -> np.ndarray:
    if windows.ndim != 3:
        raise ValueError("Windows must be a 3D array.")

    return windows.reshape(-1, windows.shape[-1])


def normalize_windows(windows: np.ndarray, baseline: int = 3, sampling_rate: int = 4) -> np.ndarray:
    baseline_range = np.arange(baseline * sampling_rate)
    eps = np.finfo(float).eps  # Small constant to prevent division by zero

    normalized_windows = []
    for window in windows:
        baseline_window = window[:, baseline_range]
        baseline_mean = np.mean(baseline_window, axis=1)
        baseline_std = np.std(baseline_window, axis=1)

        # Replace zero standard deviations with a small constant
        baseline_std = np.where(baseline_std == 0, eps, baseline_std)

        # Normalize the window
        normalized_window = (window - baseline_mean[:, None]) / baseline_std[:, None]

        # Replace any remaining invalid values (inf, nan) with zeros
        normalized_window = np.nan_to_num(normalized_window, nan=0.0, posinf=0.0, neginf=0.0)

        normalized_windows.append(normalized_window)

    normalized_windows = np.array(normalized_windows)
    return normalized_windows


class NSyncDataset:
    def __init__(self, events_data: list, signals_data: list, frame_rate: int = 30,
                 frames_averaged: int = 4, window_size: int = 21, eventlog_dict: dict | None = None,) -> None:
        self.frame_rate = frame_rate
        self.frames_averaged = frames_averaged
        self.sampling_rate = frame_rate / frames_averaged
        self.time_per_frame = 1 / frame_rate
        self.window_size = window_size
        self.window_frames = int(self.window_size * self.sampling_rate)
        self.eventlog_dict = {
            # FIXME: this dictionary is compatible with the MATLAB GUI-acquired data only
            "active_lever": 22,
            "active_lever_timeout": 222,
            "inactive_lever": 21,
            "inactive_lever_timeout": 212,
            "cue": 7,
            "infusion": 4,
        } if eventlog_dict is None else eventlog_dict

        # validating behavior data
        try:
            self.eventlog = stack_mat(events_data)
            self.event_ids, self.event_timestamps = self.eventlog[:, 0], self.eventlog[:, 1]
        except Exception as e:
            print(f"Error loading event log: {e}")

        # validating neural activity data
        try:
            self.neural_activity = signals_data
            self.extracted_signals = stack_npy(signals_data)
            self.num_neurons, self.num_frames = self.extracted_signals.shape
            self.eventlog_df = self.align_events(self.event_ids, self.event_timestamps)
        except Exception as e:
            print(f"Error loading neural activity: {e}")


    def align_events(self, event_ids: np.ndarray, event_timestamps: np.ndarray) -> np.ndarray:
        frame_indices = event_timestamps / 1000.0 / self.time_per_frame
        adjusted_timestamps = event_timestamps / 1000

        max_time = self.num_frames * self.time_per_frame
        aligned_events = np.array([event_ids, event_timestamps, adjusted_timestamps, frame_indices]).T[adjusted_timestamps <= max_time]
        return aligned_events

    def extract_unique_events(self, target_event: str = "active_lever", filter_event: str = "active_lever_timeout", sep: float = 1000.0) -> np.ndarray:
        target_events = self.eventlog_df[self.eventlog_df[:, 0] == self.eventlog_dict[target_event]]
        filter_events = self.eventlog_df[self.eventlog_df[:, 0] == self.eventlog_dict[filter_event]]

        if len(target_events) < 1:
            print("Insufficient target events found.")
            return np.array([])

        if len(filter_events) == 0:
            return target_events

        combined_events = np.sort(np.vstack((target_events, filter_events)), axis=0)

        valid_target_events = []

        for event in combined_events:
            if event[0] == self.eventlog_dict[target_event]:
                target_ts = event[1]
                filtered_ts = filter_events[:, 1]
                if not np.any(np.abs(filtered_ts - target_ts) <= sep):
                    valid_target_events.append(event)

        valid_target_events = np.array(valid_target_events) if len(valid_target_events) > 0 else np.array([])

        return valid_target_events

    def extract_normalized_windows(self, target_event: str = "active_lever") -> np.ndarray:
        if self.eventlog_df.shape[1] != 4:
            return np.array([])

        frame_indices = np.floor(
            self.eventlog_df[self.eventlog_df[:, 0] == self.eventlog_dict[target_event]][:, 3]
        ).astype(int)

        W = self.window_frames + 1
        half = W // 2

        stacked = []
        for origin in frame_indices:
            # Global start/end in the signal
            start = origin - half
            end = start + W

            # Clip to valid signal range
            sig_start = max(start, 0)
            sig_end = min(end, self.num_frames)

            # Create window of correct size
            win = np.zeros((self.num_neurons, W))

            # Calculate window indices
            win_start = sig_start - start
            win_end = win_start + (sig_end - sig_start)

            # Copy valid data
            win[:, win_start:win_end] = self.extracted_signals[:, sig_start:sig_end]
            stacked.append(win)

        return np.array(stacked)

    def mean_neural_activity(self):
        return np.mean(self.neural_activity, axis=0)

    def median_neural_activity(self):
        return np.median(self.neural_activity, axis=0)

    def raw_neural_activity(self):
        return self.neural_activity

    def eventlog_df(self):
        return self.eventlog_df