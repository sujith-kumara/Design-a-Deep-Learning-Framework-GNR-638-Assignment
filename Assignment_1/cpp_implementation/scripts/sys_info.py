import platform
import sys
import multiprocessing
import subprocess

print(f"Operating System: {platform.system()} {platform.release()}")
print(f"Python Version: {sys.version}")
print(f"CPU Cores: {multiprocessing.cpu_count()}")
try:
    if platform.system() == "Darwin":
        cpu_info = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode("utf-8").strip()
    else:
        cpu_info = subprocess.check_output(["lscpu"]).decode("utf-8")
    print(f"CPU Info: {cpu_info}")
except:
    print("Detailed CPU info not available.")
