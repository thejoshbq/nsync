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

def stack_npy(npy_list) -> np.ndarray:
    signals = []
    if isinstance(npy_list, list) and len(npy_list) > 0:
        for npy_file in npy_list:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    loaded_npy = np.load(npy_file).squeeze()
                    signals.append(loaded_npy)
                except Exception as e:
                    print(f"Error loading {npy_file}: {e}")
                    continue
                for warn in w:
                    print(f"Warning for {npy_file}: {warn.message}")
        stacked_npy = np.hstack(signals)
    else:
        stacked_npy = np.array(npy_list)

    return stacked_npy if stacked_npy.ndim == 2 else None

class NSyncDataset:
    def __init__(self, events_data: list, signals_data: list, frame_rate: int = 30,
                 frames_averaged: int = 4, window_size: int = 10, eventlog_dict: dict | None = None,) -> None:
        self.frame_rate = frame_rate
        self.frames_averaged = frames_averaged
        self.sampling_rate = frame_rate / frames_averaged
        self.time_per_frame = 1 / frame_rate
        self.window_size = window_size
        self.window_frames = int(window_size * self.sampling_rate)
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

    def extract_unique_events(self, event_type: str = "cue", filter_event: str = "active_lever_timeout", sep: int = 1000) -> np.ndarray:
        filtered_eventlog = self.eventlog_df[self.eventlog_df[:, 0] == self.eventlog_dict[event_type]]
        for idx, row in enumerate(filtered_eventlog):
            print(row)

        return filtered_eventlog

    def extract_normalized_windows(self, event_type="active_lever_timeout") -> np.ndarray:
        event_timestamps = self.eventlog_df[self.eventlog_df[:, 0] == self.eventlog_dict[event_type]][:, 1]

        pre_origin_window = int(10 * self.sampling_rate)
        event_window_size = int((pre_origin_window * 2) + (1.6 * self.sampling_rate))
        post_origin_window = event_window_size - pre_origin_window
        baseline = (0, (3 * self.sampling_rate))

        return event_timestamps



    def mean_neural_activity(self):
        return np.mean(self.neural_activity, axis=0)

    def median_neural_activity(self):
        return np.median(self.neural_activity, axis=0)

    def raw_neural_activity(self):
        return self.neural_activity

    def eventlog_df(self):
        return self.eventlog_df
