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
TODO: repeat experiments when updating the environment

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

### Doing the experiment with docker

```bash
docker run -ti -v /mnt/hdd0:/mnt/hdd0 gbarbadillo/cuda-python:python3.10-cuda14.1
cd /mnt/hdd0/MEGA/AI/22_Kaggle/arc25
pip install tqdm numpy tqdm_joblib joblib jinja2 termcolor pandas pynvml
export PYTHONPATH=/mnt/hdd0/MEGA/AI/22_Kaggle/arc25
python3 scripts/debug_parallel_execution.py

Loaded 400 tasks with 8 predictions each.
Executing predictions for batch 0 with exec: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3200/3200 [00:07<00:00, 420.28run/s]
Most common errors:
ModuleNotFoundError     1609
NonDeterministicCode     222
ValueError               105
IndexError                88
AssertionError            39
TimeoutException          29
TypeError                 11
UnsafeCode                 5
SyntaxError                4
AttributeError             4
UnboundLocalError          2
KeyError                   1
NameError                  1
Name: count, dtype: int64
      n_preds  valid code  valid outputs  unique outputs  train_pixel_score  train_correct_grids  train_pass_rate  train_is_correct  test_pixel_score  test_correct_grids  test_pass_rate  test_is_correct  is_correct
MEAN      8.0         1.0         0.3375        0.300625            0.19509             0.017259         0.011875            0.0375          0.190944            0.013594        0.013437           0.0475      0.0375

# there is a weird ModuleNotFoundError but execution is very fast
```

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

### Experiments on cluster without condor

```bash
export PYTHONPATH=/mnt/scratch/users/gbarbadillo/arc25/arc25

python3 -m venv cached-environments/debug
source cached-environments/debug/bin/activate
pip install tqdm numpy tqdm_joblib joblib jinja2 termcolor pandas pynvml

python3 arc25/scripts/debug_parallel_execution.py --dataset_path /mnt/scratch/users/gbarbadillo/arc25/data/arc-prize-2024/arc-agi_evaluation_challenges.json --prediction_path /mnt/scratch/users/gbarbadillo/arc25/predictions/2025-08-28-base-model/evaluation/8preds_2025_09_02_05_36_40_predictions.json
Traceback (most recent call last):
  File "arc25/scripts/debug_parallel_execution.py", line 5, in <module>
    from arc25.parallel_code_execution import run_code_from_predictions
  File "/mnt/scratch/users/gbarbadillo/arc25/arc25/arc25/parallel_code_execution.py", line 9, in <module>
    from arc25.code_execution import safe_code_execution
  File "/mnt/scratch/users/gbarbadillo/arc25/arc25/arc25/code_execution.py", line 133, in <module>
    def safe_code_execution(code: str, inputs: list[np.ndarray], func_name: str = 'task',
TypeError: 'type' object is not subscriptable

#This would need to use List from typing
```

## Results

## Conclusion

## Next steps

## TODO

- [ ] Can I simplify the problem so I can easily debug on the different environments?
- [ ] Maybe it could be as simple as changing the method that parallelizes the work
- [ ] Experiments I would like to do:
  - [ ] Try on laptop
  - [ ] Try on Docker
  - [ ] Try on a cluster machine without using condor
