import sys
print(f"Python version: {sys.version}")

try:
    import cupy as cp
    print('CuPy version:', cp.__version__)
    print('GPU available:', cp.cuda.is_available())
    print('Device count:', cp.cuda.runtime.getDeviceCount())
    if cp.cuda.is_available():
        meminfo = cp.cuda.runtime.memGetInfo()
        free, total = meminfo[0], meminfo[1]
        print(f'GPU Memory: Free {free/(1024**3):.2f}GB / Total {total/(1024**3):.2f}GB')
        a = cp.array([1, 2, 3])
        print('Test array on GPU:', a)
        print('Sum on GPU:', cp.sum(a))
except ImportError as e:
    print('CuPy import error:', e)
