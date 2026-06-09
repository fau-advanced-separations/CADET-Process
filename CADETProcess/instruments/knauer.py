import struct
from pathlib import Path

import numpy as np
import pandas as pd

from CADETProcess.reference import ReferenceIO

# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------


def _load_knauer_csv(file_path: str) -> tuple[dict, pd.DataFrame]:
    delimiter = ";"
    metadata = {}

    with open(file_path, encoding="latin-1") as f:
        for _ in range(6):
            line = f.readline().strip()
            if not line:
                continue
            parts = line.split(delimiter)
            key = parts[0].strip()
            value = parts[1].strip() if len(parts) > 1 else ""
            if key:
                metadata[key] = value

    data = pd.read_csv(file_path, sep=delimiter, skiprows=6, encoding="latin-1")
    data = data.dropna()

    # Normalise column names: collapse multiple spaces so "UV Channel 1  [mAU ]"
    # becomes "UV Channel 1 [mAU]"
    data.columns = [" ".join(c.split()) for c in data.columns]

    return metadata, data


# ---------------------------------------------------------------------------
# RFP binary loader
# ---------------------------------------------------------------------------

_FIRST_HEADER_OFFSET = 308   # byte offset of channel-1 block header
_CHANNEL_HEADER = 110   # per-channel header size in bytes
_EVENT_RECORD_SIZE = 108   # 8-byte float64 timestamp + 100-byte text
_COMP_PREFIX = "Composition Major Pump = "
_FLOW_PREFIX = "Flowrate Major Pump = "

# Canonical column names used internally (same as CSV after normalisation)
_RFP_SENSOR_COLS = [
    "UV Channel 1 [mAU]",
    "UV Channel 2 [mAU]",
    "UV Channel 3 [mAU]",
    "UVChannel 4 [mAU]",
    "Conductivity [mS/cm]",
    "Pressure Channel [Bar]",
]
_RFP_SOLVENT_COLS = ["Solvent A", "Solvent B", "Solvent C", "Solvent D"]


def _load_knauer_rfp(file_path: str) -> tuple[dict, pd.DataFrame]:
    with open(file_path, "rb") as f:
        data = f.read()

    # --- sensor channels ---------------------------------------------------
    sensors = {}
    hdr_off = _FIRST_HEADER_OFFSET
    num_points = None
    slice_width_ms = None

    for col in _RFP_SENSOR_COLS:
        n = struct.unpack_from("<I", data, hdr_off + 52)[0]
        sw = struct.unpack_from("<I", data, hdr_off + 56)[0]
        if num_points is None:
            num_points = n
            slice_width_ms = sw
        dat_off = hdr_off + _CHANNEL_HEADER
        sensors[col] = np.frombuffer(
            data[dat_off: dat_off + n * 4], dtype="<f4"
        )[:num_points]
        hdr_off = dat_off + n * 4

    time_array = np.arange(1, num_points + 1) * (slice_width_ms / 60_000.0)

    # --- event log ---------------------------------------------------------
    tcs_pos = data.find(b"Time Control Start")
    if tcs_pos == -1:
        raise ValueError(f"{file_path}: 'Time Control Start' not found in event log")

    ts_run_start = struct.unpack_from("<d", data, tcs_pos - 8)[0]
    n_events = struct.unpack_from("<H", data, tcs_pos - 10)[0]
    log_start = tcs_pos - 10 + 2

    composition_events: list[tuple[float, list[float]]] = []
    flow_rate_mL_min: float | None = None
    wavelengths: dict[str, int] = {}

    _WL_UV_COLS = [c for c in _RFP_SENSOR_COLS if c.startswith("UV")]

    for i in range(n_events):
        rec = log_start + i * _EVENT_RECORD_SIZE
        ts = struct.unpack_from("<d", data, rec)[0]
        text = data[rec + 8: rec + _EVENT_RECORD_SIZE].decode("latin-1").rstrip()

        if text.startswith(_COMP_PREFIX):
            run_time_min = (ts - ts_run_start) * 1440.0
            vals = [float(v.strip()) for v in text[len(_COMP_PREFIX):].split(",")]
            composition_events.append((run_time_min, vals))

        elif text.startswith(_FLOW_PREFIX) and flow_rate_mL_min is None:
            try:
                flow_rate_mL_min = float(text[len(_FLOW_PREFIX):].split()[0])
            except (ValueError, IndexError):
                pass

        elif text.startswith("Wavelength "):
            # "Wavelength N  = X nm"
            try:
                parts = text.split("=")
                idx = int(parts[0].split()[1]) - 1   # 1-based → 0-based
                wl_nm = int(parts[1].strip().split()[0])
                if 0 <= idx < len(_WL_UV_COLS):
                    wavelengths[_WL_UV_COLS[idx]] = wl_nm
            except (ValueError, IndexError):
                pass

    if not composition_events:
        raise ValueError(f"{file_path}: no 'Composition Major Pump' events found")

    comp_times = np.array([e[0] for e in composition_events])
    comp_vals = np.array([e[1] for e in composition_events])

    solvents = {}
    for j, col in enumerate(_RFP_SOLVENT_COLS):
        if len(comp_times) == 1:
            solvents[col] = np.full(num_points, comp_vals[0, j])
        else:
            solvents[col] = np.interp(time_array, comp_times, comp_vals[:, j])

    # --- binary-only metadata ----------------------------------------------
    # Operator: the field immediately after the null-terminated "Matrix Dongle
    # <id>" string in the file header.
    operator: str | None = None
    dongle_pos = data.find(b"Matrix Dongle")
    if dongle_pos != -1:
        # Skip past the dongle string (null-terminated), then skip any spaces
        null_pos = data.index(b"\x00", dongle_pos + len(b"Matrix Dongle"))
        start = null_pos + 1
        while start < null_pos + 64 and data[start:start + 1] in (b"\x00", b" "):
            start += 1
        end = data.index(b"\x00", start) if b"\x00" in data[start:start + 64] else start + 32
        operator = data[start:end].decode("latin-1").strip() or None

    metadata = {
        "Operator": operator,
        "FlowRate_mL_min": flow_rate_mL_min,
        "Wavelengths": wavelengths if wavelengths else None,
        "GradientEvents": list(zip(
            [e[0] for e in composition_events],
            [e[1] for e in composition_events],
        )),
    }

    df = pd.DataFrame({"Time [Min]": time_array, **sensors, **solvents})
    return metadata, df


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class KnauerExperimentalData:
    """
    Experimental data from a Knauer FPLC system.

    Accepts either a CSV export (``*.csv``) or a binary PurityChrom file
    (``*.rfp``).  Each data channel is exposed as a :class:`ReferenceIO`
    object.

    Parameters
    ----------
    file_path : str
        Path to the Knauer CSV or RFP file.
    flow_rate : float, optional
        Flow rate in m³/s.  Required for CSV files.  For RFP files the value
        is read from the event log when omitted.
    time_offset : float, optional
        Time offset in seconds (e.g. to account for delayed injection).
    duration : float, optional
        Duration in seconds after applying *time_offset*.  ``-1`` uses all
        data points.

    Attributes
    ----------
    uv_1, uv_2, uv_3, uv_4 : ReferenceIO or None
    conductivity : ReferenceIO or None
    pressure : ReferenceIO or None
    solvent_a, solvent_b, solvent_c, solvent_d : ReferenceIO or None
    wavelengths : dict or None
        UV wavelengths per channel (RFP only).
    operator : str or None
        Operator name from file header (RFP only).
    gradient_events : list of (float, list) or None
        Solvent composition control points from the event log as
        ``[(time_min, [A, B, C, D]), ...]``, piecewise-linearly interpolated
        by the instrument between consecutive points (RFP only).
    """

    def __init__(
        self,
        file_path: str,
        flow_rate: float | None = None,
        time_offset: float = 0.0,
        duration: float = -1,
    ) -> None:
        path = Path(file_path)

        if path.suffix.lower() == ".rfp":
            rfp_meta, self.data = _load_knauer_rfp(file_path)
            self.metadata = rfp_meta
            self.operator = rfp_meta.get("Operator")
            self.wavelengths = rfp_meta.get("Wavelengths")
            self.gradient_events = rfp_meta.get("GradientEvents")
            rfp_flow = rfp_meta.get("FlowRate_mL_min")
            if flow_rate is None:
                if rfp_flow is None:
                    raise ValueError(
                        "flow_rate not found in RFP event log; pass it explicitly."
                    )
                flow_rate = rfp_flow * 1e-6 / 60.0  # mL/min → m³/s
            self.file_name = path.stem
            self.sample = None
            self.date = None
        else:
            self.metadata, self.data = _load_knauer_csv(file_path)
            self.operator = None
            self.wavelengths = None
            self.gradient_events = None
            self.file_name = self.metadata.get("FileName")
            self.sample = self.metadata.get("Sample")
            self.date = self.metadata.get("Date")
            if flow_rate is None:
                raise ValueError("flow_rate is required for CSV files.")

        self.flow_rate = flow_rate
        self.time_offset = time_offset
        self.duration = np.inf if duration == -1 else duration

        self.uv_1 = self._make_reference("UV Channel 1 [mAU]", "uv_1")
        self.uv_2 = self._make_reference("UV Channel 2 [mAU]", "uv_2")
        self.uv_3 = self._make_reference("UV Channel 3 [mAU]", "uv_3")
        self.uv_4 = self._make_reference("UVChannel 4 [mAU]", "uv_4")
        self.conductivity = self._make_reference("Conductivity [mS/cm]", "conductivity")
        self.pressure = self._make_reference("Pressure Channel [Bar]", "pressure")
        self.solvent_a = self._make_reference("Solvent A", "solvent_a")
        self.solvent_b = self._make_reference("Solvent B", "solvent_b")
        self.solvent_c = self._make_reference("Solvent C", "solvent_c")
        self.solvent_d = self._make_reference("Solvent D", "solvent_d")

    def _make_reference(
        self,
        column_name: str,
        alias: str | None = None,
    ) -> ReferenceIO | None:
        if column_name not in self.data.columns:
            return None

        data_series = pd.to_numeric(self.data[column_name], errors="coerce")
        data_array = data_series.to_numpy().reshape(-1, 1)

        time_min = pd.to_numeric(self.data["Time [Min]"], errors="coerce").to_numpy()
        time = time_min * 60 - self.time_offset

        n = min(len(time), len(data_array))
        time = time[:n]
        data_array = data_array[:n]

        valid = (time >= 0) & (time <= self.duration)
        time = time[valid]
        data_array = data_array[valid]

        valid_data = ~np.isnan(data_array[:, 0])
        time = time[valid_data]
        data_array = data_array[valid_data]

        if len(data_array) == 0:
            return None

        name = alias if alias else column_name
        return ReferenceIO(name, time, data_array, self.flow_rate)
