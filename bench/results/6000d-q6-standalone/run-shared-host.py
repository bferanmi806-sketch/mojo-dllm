"""Run upstream run_bench without changing host power or killing other jobs.

Usage: python run-shared-host.py --config CONFIG --out RESULT [--reps 3]
Bind CUDA_VISIBLE_DEVICES to an idle allocated GPU before invocation.
"""
import argparse
import importlib.util
import json
import pathlib
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--reps', type=int, default=3)
args = parser.parse_args()
config = pathlib.Path(args.config).resolve()
output = pathlib.Path(args.out).resolve()
if json.loads(config.read_text()).get('kill_pytest', True):
    raise SystemExit('shared-host benchmark requires kill_pytest: false')
repo = pathlib.Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('official', repo / 'bench/run_bench.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
original = subprocess.run
suppressed = []


def shared_host_run(command, *positional, **keywords):
    if (isinstance(command, list) and command and command[0] == 'powerprofilesctl') or command == 'powerprofilesctl get':
        suppressed.append(command)
        return subprocess.CompletedProcess(command, 0, stdout='unchanged\n', stderr='')
    return original(command, *positional, **keywords)


subprocess.run = shared_host_run
sys.argv = [str(repo / 'bench/run_bench.py'), '--config', str(config), '--out', str(output), '--reps', str(args.reps)]
try:
    module.main()
finally:
    subprocess.run = original
data = json.loads(output.read_text())
data['machine']['power_profile'] = 'unchanged (shared-host wrapper)'
data['power_management'] = {'changed': False, 'suppressed_commands': suppressed}
output.write_text(json.dumps(data, indent=2) + '\n')
