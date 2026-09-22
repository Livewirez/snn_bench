E_MAC          = 3.7e-12   # J  (32-bit multiply-accumulate)
E_ACC          = 0.9e-12   # J  (32-bit add)
E_SRAM_READ    = 1.5e-12   # J
E_SRAM_WRITE   = 2.0e-12   # J
E_ADDRESS      = 0.5e-12   # J  (rough addressing cost per spike)

# Needed for real GPU energy
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _HAS_NVML = True
except Exception:
    _NVML_HANDLE = None
    _HAS_NVML = False