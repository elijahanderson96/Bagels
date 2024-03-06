import subprocess
import os
import signal

# List of process names to kill
process_names = ["find_best_model.py", "etf_predictor.py"]

def kill_processes(process_names):
    for process_name in process_names:
        try:
            # Find processes by name and get their PIDs
            pids = subprocess.check_output(["pgrep", "-f", process_name]).decode('utf-8').strip().split('\n')
            for pid in pids:
                os.kill(int(pid), signal.SIGKILL)
                print(f"Killed process {process_name} with PID {pid}")
        except subprocess.CalledProcessError as e:
            # No process found
            print(f"No process named {process_name} found")

kill_processes(process_names)
