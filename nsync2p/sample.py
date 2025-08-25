from pathlib import Path
import scipy.io as sio
import warnings
from tqdm import tqdm
warnings.filterwarnings('always', category=UserWarning)
warnings.filterwarnings('always', category=DeprecationWarning)

import numpy as np
from numpy.typing import NDArray
from typing import Any, Dict, List, Union

class NSyncSample:
    def __init__(
        self,
        eventlog: List[Union[str, Path]] | Dict[str, Any],
        extracted_signals: List[Union[str, Path]] | NDArray[np.uint64],
        frame_correction: Union[str, Path] | None = None,
        animal_name: str = "REX",
        target_id: Union[int, List[int]] = 22,
        noise_id: Union[int, List[int]] = 222,
        frame_rate: int = 30,
        frame_averaging: int = 4,
        isolate_events: bool = False,
        min_events: int = 3,
        normalize: bool = False,
    ):
        self._eventlog = eventlog
        self._extracted_signals = extracted_signals
        self._frame_correction = frame_correction
        self._animal_name = animal_name
        self._normalize = normalize

        if isinstance(eventlog, list):
            self._eventlog = sorted([str(f) for f in eventlog])
            self._eventlog = self.__compile_matlab_files__(eventlog)
        else:
            self._eventlog = np.array(eventlog)
        self.event_ids = self._eventlog[:, 0]
        self.event_timestamps = self._eventlog[:, 1]

        if isinstance(extracted_signals, list):
            self._extracted_signals = sorted([str(f) for f in extracted_signals])
            self._extracted_signals = self.__compile_npy_files__(extracted_signals)
        else:
            self._extracted_signals = extracted_signals.astype(np.float64)

        self.num_neurons = self._extracted_signals.shape[0]
        self.num_frames = self._extracted_signals.shape[1]

        # sample processing logic
        self._frame_rate = frame_rate
        self._frame_averaging = frame_averaging
        self._target_id = target_id
        self._noise_id = noise_id
        self._min_events = min_events
        self._isolate_events = isolate_events
        self._sampling_rate = frame_rate / frame_averaging
        self._time_per_frame = 1 / frame_rate
        self._inter_frame_interval = 1000 / self._frame_rate

        # window of interest
        self._pre_window_size = int(10 * self._sampling_rate)
        self._window_size = int((self._pre_window_size * 2) + (1.6 * self._sampling_rate))
        self._post_window_size = self._window_size - self._pre_window_size

        self._eventlog_dict = {  # defined MATLAB event IDs
            "active_lever": 22,
            "active_lever_timeout": 222,
            "inactive_lever": 21,
            "inactive_lever_timeout": 212,
            "cue": 7,
            "infusion": 4,
        }

        if self._normalize:
            self._extracted_signals = self.__normalize_signals__(self._extracted_signals)
            self._extracted_signals = self.__z_score_trace__(self._extracted_signals)

        frame_ts = self.__get_frame_timestamps__(mat_file=self._frame_correction, animal=self._animal_name)
        if self._extracted_signals.shape[1] > frame_ts.shape[0]:
            self._extracted_signals = self._extracted_signals[:, :frame_ts.shape[0] - 1]

        self._event_windows = self.__align_event_windows__()

    # private
    def __compile_matlab_files__(self, files: list) -> NDArray[np.float64]:
        stack = []
        last_timestamp = 0
        if isinstance(files, list) and len(files) > 0:
            for _, file in enumerate(tqdm(files, desc=f"{self._animal_name} | Compiling MATLAB files, n={len(files)}", total=len(files))):
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

    def __compile_npy_files__(self, files: list) -> NDArray[np.float64]:
        stack = []
        if isinstance(files, list) and len(files) > 0:
            for _, file in enumerate(tqdm(files, desc=f"{self._animal_name} | Compiling NumPy files, n={len(files)}", total=len(files))):
                with warnings.catch_warnings(record=True) as captured_warnings:
                    warnings.simplefilter("always")
                    try:
                        data = np.load(str(file)).squeeze()
                        stack.append(data)
                    except Exception as e:
                        print(f"Error loading {file}: {e}")
                        continue
                    for warning in captured_warnings:
                        if "Python 2" in str(warning.message):
                            np.save(str(file), data)
                            print(f"Resaved {file} as a Python 3 file.")
            stack = np.hstack(stack) if len(stack) > 0 else np.array(stack)
        return stack.astype(np.float64)

    @staticmethod
    def __normalize_signals__(extracted_signals: NDArray[np.float64]) -> NDArray[np.float64]:
        means = np.nanmean(extracted_signals, axis=1).reshape(-1, 1)  # shape: (num_neurons, 1)
        normalized_signals = np.divide(
            extracted_signals,
            means,
            out=np.zeros_like(extracted_signals, dtype=np.float64),
            where=(means != 0) & (~np.isnan(means))
        )

        return normalized_signals

    @staticmethod
    def __z_score_trace__(normalized_signals: NDArray[np.float64]) -> NDArray[np.float64]:
        means = np.nanmean(normalized_signals, axis=1)[:, np.newaxis]
        stds = np.nanstd(normalized_signals, axis=1)[:, np.newaxis]
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized_signals = (normalized_signals - means) / stds

        return normalized_signals

    def __get_valid_events__(self, sep: float = 1000.0) -> NDArray[np.float64]:
        target_ids = [self._target_id] if isinstance(self._target_id, int) else self._target_id
        events = np.sort(self.event_timestamps[np.isin(self.event_ids, target_ids)])

        if len(events) == 0:
            return np.array([])

        if self._isolate_events:
            events = np.delete(events, np.argwhere(np.diff(events) < sep) + 1)

        if len(events) < self._min_events:
            return np.array([])

        return events

    def __align_event_windows__(self) -> NDArray[np.float64]:
        events = self.__get_valid_events__()

        if self.get_num_frames() == 0 or self.get_num_neurons() == 0:
            raise ValueError("No valid signal data available")

        if events.ndim > 1:  # handle NaN array case for low events
            return events

        frame_indices = self.__get_event_frame_indices__(events)
        frame_indices = np.clip(frame_indices, 0, self.get_num_frames() - 1)

        num_trials = len(events)
        aligned_windows = np.full((num_trials, self._window_size, self.get_num_neurons()), np.nan, dtype=np.float64)
        signals_t = self._extracted_signals.T  # frames × neurons
        valid_trials = []
        for i, event_idx in enumerate(tqdm(frame_indices, desc=f"{self._animal_name} | Aligning event windows, n={num_trials}", total=num_trials)):
            event_idx = int(str(event_idx))
            if event_idx < self._pre_window_size or event_idx >= (self.get_num_frames() - self._post_window_size):
                continue
            aligned_windows[i] = signals_t[event_idx - self._pre_window_size: event_idx + self._post_window_size]
            valid_trials.append(i)

        if valid_trials:
            aligned_windows = aligned_windows[valid_trials]
            aligned_windows = np.swapaxes(aligned_windows, 0, 2)  # neurons × window_size × trials
        else:
            aligned_windows = np.full((self.get_num_neurons(), self._window_size, 2), np.nan, dtype=np.float64)

        return aligned_windows

    def __get_frame_timestamps__(self, mat_file: Union[str, Path] | None = None, animal: str = None) -> NDArray[
        np.float64]:
        animal = animal or self._animal_name
        special_animals = ['CTL1', 'ER-L1', 'ER-L2', 'IG-19', 'IG-28', 'PGa-T1', 'XYZ']

        frame_ts_raw = self.event_timestamps[self.event_ids == 9]

        if animal in special_animals:
            frame_ts = self.__handle_assumed_frames__(mat_file)
        else:
            frame_ts = self.__handle_missed_frames__(frame_ts_raw)

        return frame_ts[::self._frame_averaging]  # Downsample

    def __handle_assumed_frames__(self, mat_file: Union[str, Path] | None = None) -> NDArray[np.float64]:
        if mat_file is None:
            num_frames = self.get_num_frames() * self._frame_averaging
            return np.arange(num_frames) * self._inter_frame_interval
        try:
            assumed_data = sio.loadmat(mat_file)
            empty_eventlog = np.squeeze(assumed_data['eventlog'])

            max_of = np.max(empty_eventlog[:, 1])
            length_of = len(empty_eventlog[:, 1])
            x = np.vstack((empty_eventlog, empty_eventlog, empty_eventlog))
            x[length_of:, 1] += max_of
            x[2 * length_of:, 1] += 2 * max_of
            empty_eventlog = x

            frame_ts = empty_eventlog[empty_eventlog[:, 0] == 9, 1]

            dropped_frames = []
            diff_frames = np.diff(frame_ts)
            frame_drop_idx = np.where(diff_frames > 1.5 * self._inter_frame_interval)[0]
            for idx in frame_drop_idx:
                num_frames_dropped = int(
                    np.round((frame_ts[idx + 1] - frame_ts[idx]) / (self._inter_frame_interval + 0.0)) - 1)
                temp = [frame_ts[idx] + a * self._inter_frame_interval for a in range(1, num_frames_dropped + 1)]
                dropped_frames.extend(temp)
            frame_ts = np.sort(np.concatenate((frame_ts, np.array(dropped_frames))))
        except FileNotFoundError:
            num_frames = self.get_num_frames() * self._frame_averaging
            frame_ts = np.arange(num_frames) * self._inter_frame_interval

        return frame_ts

    def __handle_missed_frames__(self, frame_ts_raw: NDArray[np.float64]) -> NDArray[np.float64]:
        frame_ts = frame_ts_raw
        if frame_ts.size == 0:
            num_frames = self.get_num_frames() * self._frame_averaging
            frame_ts = np.arange(num_frames) * self._inter_frame_interval
        else:
            first_frame = np.array([0])
            last_frame = np.array([int(np.max(frame_ts) + (500 * self._inter_frame_interval))])
            frame_index_temp = np.concatenate((first_frame, frame_ts, last_frame))
            frames_missed = []
            for i in range(len(frame_index_temp) - 1):
                num_missed = int(
                    np.round((frame_index_temp[i + 1] - frame_index_temp[i]) / self._inter_frame_interval) - 1)
                if num_missed > 0:
                    for j in range(num_missed):
                        missed = frame_index_temp[i] + int(self._inter_frame_interval * (j + 1))
                        frames_missed.append(missed)
            frame_ts = np.sort(np.concatenate((frame_index_temp, frames_missed)))

        return frame_ts

    def __get_event_frame_indices__(self, events: NDArray[np.float64]) -> NDArray[np.float64]:
        frame_ts = self.__get_frame_timestamps__(mat_file=self._frame_correction, animal=self._animal_name)
        frame_indices = np.zeros(len(events), dtype=np.uint64)
        for i, event in enumerate(tqdm(events, desc=f"{self._animal_name} | Finding frame indices, n={len(events)}", total=len(events))):
            if np.isnan(event):
                frame_indices[i] = 0  # set invalid events to 0; will be filtered later
                continue
            temp = np.nonzero(frame_ts <= event)[0]
            if temp.size > 0:
                frame_indices[i] = temp[-1]
            else:
                frame_indices[i] = 0  # no frame timestamp found; use 0

        return frame_indices

    # public
    def get_animal_name(self) -> str:
        return self._animal_name

    def get_num_events(self) -> int:
        return len(self.__get_valid_events__())

    def get_min_events(self) -> int:
        return self._min_events

    def get_sampling_rate(self) -> float:
        return self._sampling_rate

    def get_event_ids(self) -> NDArray[np.float64]:
        return self.event_ids

    def get_event_timestamps(self) -> NDArray[np.float64]:
        return self.event_timestamps

    def get_num_neurons(self) -> int:
        return self.num_neurons

    def get_num_frames(self) -> int:
        return self.num_frames

    def get_window_dimensions(self) -> tuple[int, int, int]:
        return self._pre_window_size, self._post_window_size, self._window_size

    def get_eventlog(self) -> NDArray[np.float64]:
        return self._eventlog

    def get_extracted_signals(self) -> NDArray[np.float64]:
        return self._extracted_signals

    def get_event_windows(self) -> NDArray[np.float64]:
        return self._event_windows

    def __str__(self):
        return f"Animal: {self._animal_name}, n={self.num_neurons}, t={self.get_num_events()}"