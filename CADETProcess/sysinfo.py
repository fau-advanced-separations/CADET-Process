import platform

import psutil

uname = platform.uname()

system_information = {
    "system": uname.system,
    "release": uname.release,
    "machine": uname.machine,
    "processor": uname.processor,
    "n_cores": None,
    "n_cores_physical": None,
    "max_frequency": None,
    "min_frequency": None,
    "memory_total": None,
}

if psutil is not None:
    try:
        system_information["n_cores"] = psutil.cpu_count(logical=True)
        system_information["n_cores_physical"] = psutil.cpu_count(logical=False)
    except Exception:
        pass

    try:
        cpu_freq = psutil.cpu_freq()
        if cpu_freq is not None:
            system_information["max_frequency"] = cpu_freq.max
            system_information["min_frequency"] = cpu_freq.min
    except Exception:
        pass

    try:
        memory = psutil.virtual_memory()
        system_information["memory_total"] = f"{memory.total / 1024**3:.1f} GiB"
    except Exception:
        pass
