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
python3 arc25/scripts/debug_parallel_execution.py --dataset_path /mnt/scratch/users/gbarbadillo/arc25/data/arc-prize-2024/arc-agi_evaluation_challenges.json --prediction_path /mnt/scratch/users/gbarbadillo/arc25/predictions/2025-08-28-base-model/evaluation/8preds_2025_09_02_05_36_40_predictions.json --n_jobs 20 --bath_size 100

# calculon01, 12 cores
Loaded 400 tasks with 8 predictions each.
Executing predictions for batch 0 with exec: 100%|███████████████████████████████████| 3200/3200 [00:21<00:00, 150.08run/s]
Most common errors:
ModuleNotFoundError     1671
NonDeterministicCode     204
ValueError               101
IndexError                77
AssertionError            39
TimeoutException          24
TypeError                 14
AttributeError            12
SyntaxError                4
UnboundLocalError          4
UnsafeCode                 3
KeyError                   3
NameError                  1
ZeroDivisionError          1
Name: count, dtype: int64
      n_preds  valid code  valid outputs  unique outputs  ...  test_correct_grids  test_pass_rate  test_is_correct  is_correct
MEAN      8.0         1.0       0.325625        0.288438  ...            0.015469           0.015            0.055        0.05
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

### Trying to understand the problem

https://joblib.readthedocs.io/en/latest/developing.html

> The automatic array memory mapping feature of Parallel does no longer use /dev/shm if it is too small (less than 2 GB). In particular in docker containers /dev/shm is only 64 MB by default which would cause frequent failures when running joblib in Docker containers.

https://joblib.readthedocs.io/en/latest/parallel.html

Here there is a description of the Parallel method from joblib and all its parameters.

This [conversation](https://chatgpt.com/share/68cabe24-e734-8012-a409-f9e14dfa9b32) with GPT5 suggests
that signal+joblib+loky seems to be the best option.

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

