# Iteration 25. Debug parallel code execution

_16-09-2025_

## Goal

Solve the problems of slow parallel code execution in the cluster.

## Motivation

I have a method that executes code in parallel and works very fast on my computer and on Kaggle, but
weirdly it is slow on the cluster. I need to solve this issue to be able to do experiments
to validate my approach.

## Development

### Execution is slow on cluster

On my pc it executes very fast `400/400 [00:02<00:00, 152.92pred/s]` and so it is on Kaggle `960/960 [00:03<00:00, 265.18pred/s]`.

But in the cluster I'm seeing very slow executions:

- `12800/12800 [50:22<00:00,  4.24pred/s]`  [Experiment](https://wandb.ai/guillermobarbadillo/2025-09-12-search-and-learn/runs/19gni2he/logs)
- `51200/51200 [03:28<00:00, 245.29runs/s]` [Older experiment with good speed](https://wandb.ai/guillermobarbadillo/2025-09-07-search-and-learn/runs/zdkkfzdv/logs)

#### Reverting to old code

I have tried reverting back to commit `1557726a0e184d1a4e0b0490eec44bde7dde304e`, from 8 september when I logged fast execution times. However the problem persisted:

- 4 cpus -> 41.67runs/s
- 8 cpus -> 61.31runs/s
- 20 cpus -> 56.75runs/s
- 64 cpus -> 9.51runs/s
- 128 cpus -> 9.41 runs/s

I have also tried running on other machine (calculon19 instead of calculon21) but did not get better results:

- 8 -> 74.22runs/s
- 16 -> 86.01runs/s

#### Simpler script

Iterations have been slow because I'm doing inference with the model first. That makes that each
execution takes around 30 minutes. I need to create a script that allows me to see results much faster.
That way I will run the same script with the same data in the different settings and get more information
about the problem faster.

I have prepared the script and I cannot understand the problem. Could it be a problem with the environment?

```bash
python scripts/debug_parallel_execution.py
Loaded 400 tasks with 8 predictions each.
Executing predictions for batch 0 with exec: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3200/3200 [00:09<00:00, 347.91run/s]
Most common errors:
NonDeterministicCode    222
ValueError              214
IndexError              203
AssertionError          181
TimeoutException         49
AttributeError           21
TypeError                17
UnboundLocalError         7
UnsafeCode                5
KeyError                  4
SyntaxError               4
StopIteration             4
NameError                 4
ZeroDivisionError         2
RecursionError            1
Name: count, dtype: int64
      n_preds  valid code  valid outputs  unique outputs  train_pixel_score  train_correct_grids  train_pass_rate  train_is_correct  test_pixel_score  test_correct_grids  test_pass_rate  test_is_correct  is_correct
MEAN      8.0         1.0       0.706875        0.629062           0.415273             0.022264          0.01375            0.0425          0.404596            0.016719        0.016563             0.06        0.04


export N_CPUS=8; condor_submit train.condor command=" python /mnt/scratch/users/gbarbadillo/arc25/arc25/scripts/debug_parallel_execution.py \ --dataset-path /mnt/scratch/users/gbarbadillo/arc25/data/arc-prize-2024/arc-agi_evaluation_challenges.json \ --prediction-path /mnt/scratch/users/gbarbadillo/arc25/predictions/2025-08-28-base-model/evaluation/8preds_2025_09_02_05_36_40_predictions.json" -append request_cpus=${N_CPUS} -append request_gpus=0
```

Notice that I get fast execution on my PC also if I disable memmapping: `max_nbytes=None, mmap_mode=None)`.
The speed seems to be the same, so probably wasn't using previously because arrays are very small.

### Doing the experiment with docker on my machine

```bash
docker run -ti -v /mnt/hdd0:/mnt/hdd0 gbarbadillo/cuda-python:python3.10-cuda14.1
cd /mnt/hdd0/MEGA/AI/22_Kaggle/arc25
pip install tqdm numpy tqdm_joblib joblib jinja2 termcolor pandas pynvml scipy
export PYTHONPATH=/mnt/hdd0/MEGA/AI/22_Kaggle/arc25
python3 scripts/debug_parallel_execution.py

Loaded 400 tasks with 8 predictions each.
Executing predictions for batch 0 with exec: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3200/3200 [00:09<00:00, 354.43run/s]
Most common errors:
NonDeterministicCode    222
ValueError              214
IndexError              202
AssertionError          181
TimeoutException         49
AttributeError           21
TypeError                17
UnboundLocalError         7
UnsafeCode                5
KeyError                  4
SyntaxError               4
StopIteration             4
NameError                 4
ZeroDivisionError         2
RecursionError            1
Name: count, dtype: int64
      n_preds  valid code  valid outputs  unique outputs  train_pixel_score  train_correct_grids  train_pass_rate  train_is_correct  test_pixel_score  test_correct_grids  test_pass_rate  test_is_correct  is_correct
MEAN      8.0         1.0       0.707187        0.629375           0.415056             0.022014          0.01375            0.0425          0.404481            0.016719        0.016563             0.06        0.04

# I can restrict shm size and speed is not affected
docker run -ti --shm-size=64M -v /mnt/hdd0:/mnt/hdd0 gbarbadillo/cuda-python:python3.10-cuda14.1
353.00run/s
```

Runs as fast as without docker when using my machine.

### Experiments on laptop

```bash
export PYTHONPATH=/mnt/data/other/code/arc25
python scripts/debug_parallel_execution.py --dataset_path /mnt/scratch/users/gbarbadillo/arc25/data/arc-prize-2024/arc-agi_evaluation_challenges.json --prediction_path /mnt/scratch/users/gbarbadillo/arc25/predictions/2025-08-28-base-model/evaluation/8preds_2025_09_02_05_36_40_predictions.json

Loaded 400 tasks with 8 predictions each.
Executing predictions for batch 0 with exec: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3200/3200 [00:12<00:00, 258.22run/s]
Most common errors:
ModuleNotFoundError     1671
NonDeterministicCode     204
ValueError               107
IndexError                77
AssertionError            39
TimeoutException          34
TypeError                 14
AttributeError            12
UnboundLocalError          4
SyntaxError                4
UnsafeCode                 3
KeyError                   3
NameError                  1
ZeroDivisionError          1
Name: count, dtype: int64
      n_preds  valid code  valid outputs  unique outputs  train_pixel_score  train_correct_grids  train_pass_rate  train_is_correct  test_pixel_score  test_correct_grids  test_pass_rate  test_is_correct  is_correct
MEAN      8.0         1.0       0.320625        0.285938           0.184007             0.016169         0.010937            0.0475          0.180193            0.013281        0.012812           0.0525      0.0475
```

Again, this is very fast.

### Experiments on cluster without condor or docker

```bash
# create environment
python3 -m venv cached-environments/debug
source cached-environments/debug/bin/activate
pip install tqdm numpy tqdm_joblib joblib jinja2 termcolor pandas pynvml scipy

# launch script
source cached-environments/debug/bin/activate
export PYTHONPATH=/mnt/scratch/users/gbarbadillo/arc25/arc25
python3 arc25/scripts/debug_parallel_execution.py --dataset_path /mnt/scratch/users/gbarbadillo/arc25/data/arc-prize-2024/arc-agi_evaluation_challenges.json --prediction_path /mnt/scratch/users/gbarbadillo/arc25/predictions/2025-08-28-base-model/evaluation/8preds_2025_09_02_05_36_40_predictions.json --n_jobs 20

# calculon01, 12 cores
Loaded 400 tasks with 8 predictions each.
Executing predictions for batch 0 with exec: 100%|████████████████████████████████████| 3200/3200 [00:36<00:00, 86.68run/s]
Most common errors:
NonDeterministicCode    204
ValueError              204
IndexError              184
AssertionError          168
TimeoutException         89
TypeError                27
AttributeError           25
UnboundLocalError         9
KeyError                  5
SyntaxError               4
ZeroDivisionError         4
UnsafeCode                3
NameError                 3
StopIteration             2
Name: count, dtype: int64
      n_preds  valid code  valid outputs  unique outputs  ...  test_correct_grids  test_pass_rate  test_is_correct  is_correct
MEAN      8.0         1.0       0.709063            0.63  ...            0.019375         0.01875             0.07      0.0
# not bad, considering that it had other workloads at the same time

## calculon18, 64 cores
n_jobs=-1, 114.21run/s]
n_jobs=2, 82.98run/s
n_jobs=5, 153.94run/s
n_jobs=10, 171.64run/s
n_jobs=20, 151.97run/s
n_jobs=60, 140.57run/s
n_jobs=-1, 140.47run/s
# there is like a big startup time that does not happen on my machine

## calculon13, 20 cores
n_jobs=-1, 222.78run/s
n_jobs=5, 175.63run/s
n_jobs=10, 206.38run/s
n_jobs=20, 238.54run/s
# there is a weird startup time, and sometimes ending time

## calculon21, 252cores
Loaded 400 tasks with 8 predictions each.
Executing predictions for batch 0 with exec: 100%|████████████████████████████████████| 3200/3200 [00:36<00:00, 87.61run/s]
# It might be the problem of the machine. Need to try on different machines with different number of cores
# Notice that Kaggle machines have 48 cores. https://cloud.google.com/compute/docs/gpus#l4-gpus

n_jobs=-1, 138.33run/s
n_jobs=5, 162.16run/s
n_jobs=10, 172.60run/s
n_jobs=20, 180.29run/s
```

### Experiments on cluster with docker

```bash

sudo sudo docker run -ti -v /mnt/scratch/users/gbarbadillo/arc25:/mnt/scratch/users/gbarbadillo/arc25 gbarbadillo/cuda-python:python3.10-cuda14.1
cd /mnt/scratch/users/gbarbadillo/arc25
source cached-environments/venv_0e8c9c65f4e428eaa5db41171ac52335/bin/activate
export PYTHONPATH=/mnt/scratch/users/gbarbadillo/arc25/arc25
python3 arc25/scripts/debug_parallel_execution.py --dataset_path /mnt/scratch/users/gbarbadillo/arc25/data/arc-prize-2024/arc-agi_evaluation_challenges.json --prediction_path /mnt/scratch/users/gbarbadillo/arc25/predictions/2025-08-28-base-model/evaluation/8preds_2025_09_02_05_36_40_predictions.json --n_jobs 20

## calculon21, 252cores
Executing predictions for batch 0 with exec:  13%|████▋                                | 401/3200 [04:26<30:57,  1.51run/s]
n_jobs=20, 1.51run/s

# try increasing shm
sudo sudo docker run -ti --shm-size=2g -v /mnt/scratch/users/gbarbadillo/arc25:/mnt/scratch/users/gbarbadillo/arc25 gbarbadillo/cuda-python:python3.10-cuda14.1

## calculon21, 252cores
n_jobs=2, 1.25run/s
n_jobs=5, 1.48run/s
n_jobs=20, 1.55run/s


# try using /dev/shm
sudo docker run -ti --ipc=host --shm-size=2g \
  -e TMPDIR=/dev/shm -e JOBLIB_TEMP_FOLDER=/dev/shm -e LOKY_TEMP=/dev/shm \
  -v /mnt/scratch/users/gbarbadillo/arc25:/mnt/scratch/users/gbarbadillo/arc25 \
  gbarbadillo/cuda-python:python3.10-cuda14.1
n_jobs=20, 1.39run/s

### confirm cpu limits inside the docker
sudo sudo docker run -ti -v /mnt/scratch/users/gbarbadillo/arc25:/mnt/scratch/users/gbarbadillo/arc25 gbarbadillo/cuda-python:python3.10-cuda14.1
# cgroup v2 (common on modern kernels)
$ cat /sys/fs/cgroup/cpu.max 2>/dev/null || true
$ # cgroup v1 (older)
$ cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || true
-1
$ cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || true
100000
$ 
$ # Are we being throttled?
$ cat /sys/fs/cgroup/cpu.stat 2>/dev/null || cat /sys/fs/cgroup/cpu/cpu.stat 2>/dev/null || true
nr_periods 0
nr_throttled 0
throttled_time 0
$ 
$ # How many CPUs are we actually allowed to run on?
$ grep Cpus_allowed_list /proc/self/status
Cpus_allowed_list:	0-255
$ nproc
256

## I have also tried setting environment flags without success
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export TMPDIR=/dev/shm JOBLIB_TEMP_FOLDER=/dev/shm LOKY_TEMP=/dev/shm
```

This shows that the problem only happens when using docker on the cluster. Docker has access to all the cpus, 
we can set a big enough shm size, we can disable memmapping, but the execution is still slow.

### The problem seems to be related to the environment!

#### Experiments on cluster with different environments

```bash
sudo sudo docker run -ti -v /mnt/scratch/users/gbarbadillo/arc25:/mnt/scratch/users/gbarbadillo/arc25 gbarbadillo/cuda-python:python3.10-cuda14.1
cd /mnt/scratch/users/gbarbadillo/arc25
export PYTHONPATH=/mnt/scratch/users/gbarbadillo/arc25/arc25
pip install tqdm numpy tqdm_joblib joblib jinja2 termcolor pandas pynvml scipy

python3 arc25/scripts/debug_parallel_execution.py --dataset_path /mnt/scratch/users/gbarbadillo/arc25/data/arc-prize-2024/arc-agi_evaluation_challenges.json --prediction_path /mnt/scratch/users/gbarbadillo/arc25/predictions/2025-08-28-base-model/evaluation/8preds_2025_09_02_05_36_40_predictions.json --n_jobs 20
# 307.04run/s

Executing predictions for batch 0 with exec: 100%|███████████████████████████████████| 3200/3200 [00:10<00:00, 307.04run/s]
Most common errors:
ValueError              211
NonDeterministicCode    204
IndexError              186
AssertionError          172
TimeoutException         42
TypeError                28
AttributeError           25
UnboundLocalError         9
KeyError                  5
SyntaxError               4
ZeroDivisionError         4
UnsafeCode                3
NameError                 3
StopIteration             2
Name: count, dtype: int64
      n_preds  valid code  valid outputs  unique outputs  ...  test_correct_grids  test_pass_rate  test_is_correct  is_correct
MEAN      8.0         1.0       0.719375         0.63875  ...            0.020313        0.019688           0.0725      0.0575

# however if I activate the environment
source cached-environments/venv_0e8c9c65f4e428eaa5db41171ac52335/bin/activate
python3 arc25/scripts/debug_parallel_execution.py --dataset_path /mnt/scratch/users/gbarbadillo/arc25/data/arc-prize-2024/arc-agi_evaluation_challenges.json --prediction_path /mnt/scratch/users/gbarbadillo/arc25/predictions/2025-08-28-base-model/evaluation/8preds_2025_09_02_05_36_40_predictions.json --n_jobs 20
# 1.11s/run

# create a new environment
deactivate
python3 -m venv cached-environments/debug-2
source cached-environments/debug-2/bin/activate
pip install tqdm numpy tqdm_joblib joblib jinja2 termcolor pandas pynvml scipy
python3 arc25/scripts/debug_parallel_execution.py --dataset_path /mnt/scratch/users/gbarbadillo/arc25/data/arc-prize-2024/arc-agi_evaluation_challenges.json --prediction_path /mnt/scratch/users/gbarbadillo/arc25/predictions/2025-08-28-base-model/evaluation/8preds_2025_09_02_05_36_40_predictions.json --n_jobs 20
# 127.53run/s
```

These results show that there is something wrong with `venv_0e8c9c65f4e428eaa5db41171ac52335` that makes execution very slow. Notice that the filesystem doesn't seem to be the problem because we have created a new environment
on the same folder and it is much faster.

One weird thing is that they all have the same versions of the python libraries:

```bash
python3 - <<'PY'
import joblib, numpy, scipy, pandas

print("joblib:", joblib.__version__)
print("numpy:", numpy.__version__)
print("scipy:", scipy.__version__)
print("pandas:", pandas.__version__)
from joblib.externals import loky
print("loky version:", getattr(loky, "__version__", "vendored"))
PY


joblib: 1.5.2
numpy: 2.2.6
scipy: 1.15.3
pandas: 2.3.2
loky version: 3.5.6
```

Other tests suggested by GPT5 also show the same outputs:

```bash
python3 - <<'PY'
import os, tempfile, glob, time, joblib, numpy as np
print("which python:", os.popen("which python").read().strip())
print("TMPDIR:", os.getenv("TMPDIR"))
print("JOBLIB_TEMP_FOLDER:", os.getenv("JOBLIB_TEMP_FOLDER"))
print("LOKY_TEMP:", os.getenv("LOKY_TEMP"))
print("loky dirs now:", glob.glob("/dev/shm/loky-*")[:3])

from joblib import Parallel, delayed
def f(x): return x*x

start=time.time()
Parallel(n_jobs=20, backend="loky", batch_size="auto")(delayed(f)(i) for i in range(20000))
print("Parallel microbench elapsed:", round(time.time()-start,3), "s")
PY

which python: 
TMPDIR: None
JOBLIB_TEMP_FOLDER: None
LOKY_TEMP: None
loky dirs now: []
Parallel microbench elapsed: 2.437 s

which python: /mnt/scratch/users/gbarbadillo/arc25/cached-environments/debug-2/bin/python
TMPDIR: None
JOBLIB_TEMP_FOLDER: None
LOKY_TEMP: None
loky dirs now: []
Parallel microbench elapsed: 4.954 s

which python: /mnt/scratch/users/gbarbadillo/arc25/cached-environments/venv_0e8c9c65f4e428eaa5db41171ac52335/bin/python
TMPDIR: None
JOBLIB_TEMP_FOLDER: None
LOKY_TEMP: None
loky dirs now: []
Parallel microbench elapsed: 80.421 s
```

```bash
python3 - <<'PY'
import os, multiprocessing as mp, joblib
print("python:", os.popen("which python").read().strip())
print("mp start method:", mp.get_start_method(allow_none=True))
# joblib/loky usually uses 'loky' context internally, but this reveals if something forced 'spawn'
import joblib.externals.loky.backend.context as lctx
print("loky default context:", lctx.get_context().get_start_method())
PY

They are all equal:

python: 
mp start method: None
loky default context: loky

python: /mnt/scratch/users/gbarbadillo/arc25/cached-environments/debug-2/bin/python
mp start method: None
loky default context: loky

python: /mnt/scratch/users/gbarbadillo/arc25/cached-environments/venv_0e8c9c65f4e428eaa5db41171ac52335/bin/python
mp start method: None
loky default context: loky
```

However one weird thing is that launching the python terminal is much slower in
the slow environment.

```bash
time python -c "pass"
# The slow environment takes 4s to start, the fast environment 0.5s
```

Let's try to find the root of the problem

```bash
sudo sudo docker run -ti -v /mnt/scratch/users/gbarbadillo/arc25:/mnt/scratch/users/gbarbadillo/arc25 gbarbadillo/cuda-python:python3.10-cuda14.1
cd /mnt/scratch/users/gbarbadillo/arc25
export PYTHONPATH=/mnt/scratch/users/gbarbadillo/arc25/arc25
source cached-environments/venv_0e8c9c65f4e428eaa5db41171ac52335/bin/activate
source cached-environments/debug-2/bin/activate

```

What if I create a big environment? Could the problem be size-related?

```bash
python3 -m venv cached-environments/debug-big
source cached-environments/debug-big/bin/activate
pip install -r arc25/requirements.txt
export PYTHONPATH=/mnt/scratch/users/gbarbadillo/arc25/arc25
python3 arc25/scripts/debug_parallel_execution.py --dataset_path /mnt/scratch/users/gbarbadillo/arc25/data/arc-prize-2024/arc-agi_evaluation_challenges.json --prediction_path /mnt/scratch/users/gbarbadillo/arc25/predictions/2025-08-28-base-model/evaluation/8preds_2025_09_02_05_36_40_predictions.json --n_jobs 20
# ~1 run/s
```



#### Recreate environment at home PC

Let's see if recreating the environment at home results on slow execution.

```bash
docker run -ti -v /mnt/hdd0:/mnt/hdd0 gbarbadillo/cuda-python:python3.10-cuda14.1
python3 -m venv /mnt/hdd0/TEMP/cached-environment-from-requirements
source /mnt/hdd0/TEMP/cached-environment-from-requirements/bin/activate
cd /mnt/hdd0/MEGA/AI/22_Kaggle/arc25
pip install -r requirements.txt
export PYTHONPATH=/mnt/hdd0/MEGA/AI/22_Kaggle/arc25
python3 scripts/debug_parallel_execution.py
#267-350 run/s

python3 -m venv /mnt/hdd0/TEMP/cached-environment-simple
source /mnt/hdd0/TEMP/cached-environment-simple/bin/activate
pip install tqdm numpy tqdm_joblib joblib jinja2 termcolor pandas nvidia-ml-py scipy
export PYTHONPATH=/mnt/hdd0/MEGA/AI/22_Kaggle/arc25
python3 scripts/debug_parallel_execution.py
# 331-350 run/s
```

On my PC I can't replicate the problem, at least with the latest version of the requirements execution
time is fast in both cases.

### Simplying the code to diagnose the problem

This could be a minimal code snippet to reproduce the problem.

```bash
python3 - <<'PY'
import time
import sys
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
print('Starting parallel microbench...')
def f(x): return x*x
start = time.time()
n = 20000
with tqdm_joblib(total=n):
    Parallel(n_jobs=20, backend="loky", batch_size="auto")(delayed(f)(i) for i in range(n))
execution_time = time.time() - start
print(f"Parallel microbench elapsed {round(execution_time,3)}s for python path: {sys.executable}")
PY
time python -c "pass"
```

The following lines are useful to run the tests in different environments

```bash
sudo sudo docker run -ti -v /mnt/scratch/users/gbarbadillo/arc25:/mnt/scratch/users/gbarbadillo/arc25 gbarbadillo/cuda-python:python3.10-cuda14.1
source /mnt/scratch/users/gbarbadillo/arc25/cached-environments/venv_0e8c9c65f4e428eaa5db41171ac52335/bin/activate
source /mnt/scratch/users/gbarbadillo/arc25/cached-environments/debug/bin/activate
source /mnt/scratch/users/gbarbadillo/arc25/cached-environments/debug-2/bin/activate
source /mnt/scratch/users/gbarbadillo/arc25/cached-environments/debug-big/bin/activate
```

The more the workers the higher the benchmark time. Maybe one work around would be to adjust the number
of workers to the number of predictions. For example use a single worker when the number of tasks is small.

```
# calculon21
njobs, benchmark time
1, 0.105
2, 15.763
4, 21.011
8, 33,5
20, 80.97
```

Let's rewrite the code.

```bash
python3 - <<'PY'
import time
import sys
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
start = time.time()
parallel = Parallel(n_jobs=20, backend="loky", batch_size="auto")
execution_time = time.time() - start
print(f"Creating the parallel object took {round(execution_time,3)}s")
print('Starting parallel microbench...')
def f(x): return x*x
start = time.time()
n = 20000
with tqdm_joblib(total=n):
    parallel(delayed(f)(i) for i in range(n))
execution_time = time.time() - start
print(f"Parallel microbench elapsed {round(execution_time,3)}s for python path: {sys.executable}")
start = time.time()
n = 20000
with tqdm_joblib(total=n):
    parallel(delayed(f)(i) for i in range(n))
execution_time = time.time() - start
print(f"Parallel microbench second round elapsed {round(execution_time,3)}s for python path: {sys.executable}")
PY

Creating the parallel object took 0.0s
Starting parallel microbench...
100%|███████████████████████████████████████████████████████████████████████████████| 20000/20000 [00:59<00:00, 337.62it/s]
Parallel microbench elapsed 59.242s for python path: /mnt/scratch/users/gbarbadillo/arc25/cached-environments/venv_0e8c9c65f4e428eaa5db41171ac52335/bin/python3
100%|█████████████████████████████████████████████████████████████████████████████| 20000/20000 [00:00<00:00, 37853.86it/s]
Parallel microbench second round elapsed 0.529s for python path: /mnt/scratch/users/gbarbadillo/arc25/cached-environments/ven
```

This shows that if we reuse the parallel object the second run is really fast. That is the solution to our problem.
And that explains why runs where fast when I was not using batches in the cluster. 

### Trying to understand the problem

https://joblib.readthedocs.io/en/latest/developing.html

> The automatic array memory mapping feature of Parallel does no longer use /dev/shm if it is too small (less than 2 GB). In particular in docker containers /dev/shm is only 64 MB by default which would cause frequent failures when running joblib in Docker containers.

https://joblib.readthedocs.io/en/latest/parallel.html

Here there is a description of the Parallel method from joblib and all its parameters.

This [conversation](https://chatgpt.com/share/68cabe24-e734-8012-a409-f9e14dfa9b32) with GPT5 suggests
that signal+joblib+loky seems to be the best option.

### Explanation of the problem

When we use joblib and loky to parallelize python execution, it creates n workers. 
I don't know the reason, but in the cluster when we use a big python environment, the creation of a python environment is slow. It can take 3-5 seconds or even more.

Thus if we span 20 workers, it will take 60 seconds (3*20) to create those workers. The more the workers the bigger the startup time. That does not happen on my machine or Kaggle, it is almost instantenous to span workers. It is probably related to using a slow distributed filesystem.

On my first implementation I run all the jobs at once, and for tasks that took
around 10-20 minutes to execute the startup time was not important. 

But my second implementation used smaller batches, that could take 10-15 seconds to run. In that case the startup time dominates.

What is the solution? I have to reuse the parallel object between different runs. That way I only pay the startup time once (If there are execution errors I might have to regenerate the object, but that's a separate issue). Thus I have to encapsulate the execution function inside an object, that stores the parallel object.

Why solving this problem was so difficult? Because there could be a lot of possible causes:

- Changes in the environment
- Changes in the code
- Changes in the cluster
- Problems with cluster disk 
- At the beginning iteration was very slow because it was coupled with inference
- The problem happened only on the cluster, making testing more difficult
- Different joblib parameters

### The problem is still not solved

These logs from RL fine-tuning show that the problem is not solved. Executing the reward function
should be almost instantaneous:

```
2025-09-19 09:14:29,893 - arc25.logging - INFO - wrapper - Executed arc_reward in 98.5315 seconds
  0%|          | 1/40000 [02:20<1557:46:20, 140.20s/it]2025-09-19 09:15:09,095 - arc25.logging - INFO - wrapper - Executing arc_reward...
2025-09-19 09:15:29,255 - __main__ - INFO - arc_reward - Completions length: [379, 268, 290, 261, 258, 426, 529, 230, 321, 411, 315, 416, 369, 215, 222, 293, 269, 407, 430, 274, 603, 277, 303, 269, 329, 425, 528, 460, 287, 347, 383, 430]
2025-09-19 09:15:29,256 - __main__ - INFO - arc_reward - Rewards: [1.0171387073347857, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.194082455235095, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.3031590413943355, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
2025-09-19 09:15:29,257 - arc25.logging - INFO - wrapper - Executed arc_reward in 20.1615 seconds
  0%|          | 2/40000 [03:27<1079:56:44, 97.20s/it]2025-09-19 09:16:11,448 - arc25.logging - INFO - wrapper - Executing arc_reward...
2025-09-19 09:16:12,491 - __main__ - INFO - arc_reward - Completions length: [321, 317, 292, 372, 312, 309, 265, 290, 310, 363, 289, 324, 333, 248, 359, 223, 334, 393, 305, 307, 356, 451, 288, 308, 350, 284, 363, 318, 318, 269, 401, 313]
2025-09-19 09:16:12,492 - __main__ - INFO - arc_reward - Rewards: [0.0, 1.540509259259259, 1.2694444444444444, 1.5388888888888888, 0.0, 10.0, 1.25, 1.25, 1.2694444444444444, 1.5447089947089947, 1.5849074074074072, 1.3285069444444444, 0.0, 1.413287037037037, 0.0, 4.4944444444444445, 1.6450396825396827, 1.2327548912075497, 1.4410404410404412, 0.0, 1.413287037037037, 0.0, 0.0, 1.8173611111111114, 1.381875, 0.0, 1.5, 1.4372685185185186, 1.0869675925925926, 1.1830555555555555, 1.5, 1.8173611111111114]
2025-09-19 09:16:12,493 - arc25.logging - INFO - wrapper - Executed arc_reward in 1.0444 seconds
  0%|          | 3/40000 [03:48<694:53:10, 62.54s/it]2025-09-19 09:16:40,261 - arc25.logging - INFO - wrapper - Executing arc_reward...
2025-09-19 09:17:09,099 - __main__ - INFO - arc_reward - Completions length: [459, 472, 536, 394, 502, 450, 371, 762, 368, 380, 525, 482, 339, 439, 379, 353, 415, 528, 317, 469, 541, 425, 326, 488, 424, 447, 421, 443, 391, 419, 422, 482]
2025-09-19 09:17:09,101 - __main__ - INFO - arc_reward - Rewards: [1.6681249999999999, 0.0, 0.0, 1.65875, 0.0, 0.0, 0.0, 0.0, 0.0, 1.9325, 0.0, 0.0, 1.92, 0.0, 0.0, 1.758125, 1.7893750000000002, 0.0, 1.9325, 0.0, 1.788125, 0.0, 1.930625, 0.0, 0.0, 1.67625, 0.0, 0.0, 0.0, 1.87625, 0.0, 0.0]
2025-09-19 09:17:09,103 - arc25.logging - INFO - wrapper - Executed arc_reward in 28.8413 seconds
  0%|          | 4/40000 [05:04<752:27:40, 67.73s/it]2025-09-19 09:17:47,714 - arc25.logging - INFO - wrapper - Executing arc_reward...
2025-09-19 09:18:36,878 - __main__ - INFO - arc_reward - Completions length: [220, 353, 217, 287, 244, 333, 245, 221, 229, 227, 283, 226, 221, 220, 222, 260, 222, 196, 237, 311, 230, 231, 259, 325, 409, 210, 257, 268, 397, 224, 304, 278]
2025-09-19 09:18:36,879 - __main__ - INFO - arc_reward - Rewards: [0.0, 0.0, 0.0, 0.0, 0.0, 1.79, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.79, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.79, 0.0]
2025-09-19 09:18:36,881 - arc25.logging - INFO - wrapper - Executed arc_reward in 49.1656 seconds
```

I need to check the use of tar.gz files for the python environment that are copied to the machine
at the start of the run.

## Results

## Conclusion

## Next steps

## TODO

- [x] Can I simplify the problem so I can easily debug on the different environments?
- [ ] Maybe it could be as simple as changing the method that parallelizes the work
- [x] Experiments I would like to do:
  - [x] Try on laptop
  - [x] Try on Docker
  - [x] Try on a cluster machine without using condor
  - [x] Try on a cluster machine with docker but without condor

