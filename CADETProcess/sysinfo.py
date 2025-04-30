import platform

import psutil

uname = platform.uname()
cpu_freq = psutil.cpu_freq()
memory = psutil.virtual_memory()
memory_total = psutil._common.bytes2human(memory.total)

system_information = {
    "system": uname.system,
    "release": uname.release,
    "machine": uname.machine,
    "processor": uname.processor,
    "n_cores": psutil.cpu_count(logical=True),
    "n_cores_physical": psutil.cpu_count(logical=False),
    "max_frequency": cpu_freq.max,
    "min_frequency": cpu_freq.min,
    "memory_total": memory_total,
}
