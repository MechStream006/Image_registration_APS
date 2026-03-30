import os
import cupy as cp

# Add CuPy's DLL path to system PATH
cupy_path = os.path.dirname(cp.__file__)
dll_path = os.path.join(cupy_path, '.data')

if os.path.exists(dll_path):
    os.add_dll_directory(dll_path)
    print(f"Added CuPy DLL path: {dll_path}")