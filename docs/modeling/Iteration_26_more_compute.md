# Iteration 26. Acquire more compute

_23-09-2025_

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.

<details>
  <summary>Click to expand/collapse this section</summary>
</details>
--->

## Goal

Acquire more compute to be able to do experiments before the end of the competition

## Motivation

Currently we have 13xA6000 GPUs and 2xH100 GPUs in Veridas' cluster. However we are going to loose
half of the A6000 due to problems with the machine so competition for the hardware will be fierce.

It's time to use the compute offered by ARC prize partners to be able to do my experiments.

## Development

https://arcprize.org/partners

- [x] Strong Compute. $5k-$50k. 
- [x] Lambda. $1K. Applied and received 1k in compute credits.
- [ ] Google. $?. Applied, waiting for approval
- [ ] Hyperbolic. $1K. . Applied, waiting for approval
- [ ] Modal. $500. Already applied, waiting for approval
- RunPod. $100. Decided not to apply.

### Strong Compute

I did the introductory call on 29/09/2025. My initial intention is to use machines with 3090 to run
experiments with search and learn and find the best possible configuration.

<https://cp.strongcompute.ai/>

Notes from meeting with Adam:

- [Documentation](https://docs.strongcompute.com/)
  - Burst is for AI Model training. I can see many burst experiments from last year's ARC edition.
- **Datasets**. I can use datasets to store models or data. They can be created from S3 objects or Huggingface. They say it should be faster than using Huggingface directly. At least I don't have to download the data to the temporal folder.
- **Shapes**. I can see the available machine types.
- I need wiregard to connect to the Sydney cluster
- They can help with multi-node trainings

#### Logbook

- I have created a new project for the experiments
- I have started a container on `Sydney Strong Compute Cluster` with 0 GPUs.
- I have added [wireguard](https://docs.strongcompute.com/~/revisions/amxNDLwVFA8r5isl8NAV/getting-started/vpn-sydney-cluster-only) and configured it: `sudo wg-quick up wg0` to connect and `sudo wg-quick down wg0` to disconnect.
- I'm following the [Hello world](https://docs.strongcompute.com/getting-started/2.-hello-world-training-example) guide.
- Documentation is very complete. It might take a while to get used and started but once I launch the
first run I believe it will be very fast to launch multiple experiments.
- References from previous arc edition:
  - https://github.com/ironbar/arc24/tree/main/scripts/strong_compute
  - https://ironbar.github.io/arc24/modeling/Iteration_44_learn_to_use_strong_compute/
- I was losing internet connection when using Wireguard, removing the line with `DNS = ` fixed it. [ChatGPT](https://chatgpt.com/c/68da8cb6-22f8-8328-9388-b559f2872c8c)
- Create a key with `ssh-keygen` and add it to github. Remember to delete it once the challenge is over.
- Install the requirements, I believe flash-attn requires a machine with GPU to be installed

```bash
apt update && apt install -y python3-dev python3-pip python3-virtualenv git nano vim
git config --global user.name "Guillermo Barbadillo"
git config --global user.email "guillermobarbadillo@gmail.com"
cd ~
git clone git@github.com:ironbar/arc25.git
python3 -m virtualenv ~/arc25_env
source ~/arc25_env/bin/activate
pip install -r arc25/requirements.txt
pip install unsloth_zoo==2025.9.6 # I should update the requirements
MAX_JOBS=2 python -m pip install flash-attn==2.6.3 --no-build-isolation

vim secrets.sh #export WANDB_API_KEY=
chmod +x secrets.sh
```

- Created dataset from huggingface [barc0/Llama-3.1-ARC-Potpourri-Induction-8B](https://huggingface.co/barc0/Llama-3.1-ARC-Potpourri-Induction-8B)
  I don't know why but the first day it was not available, the second day it was. Seems to take time to generate the dataset.
- After installing the requirements stopping the machine took more time, probably due to saving the environment
- Starting a job takes around 10 minutes (probably spend copying the environment)
- First running job failed after 14 minutes without any error message. Might be related to low disk space, maybe I should change the cache directory for huggingface.

#### Problems with unsloth

When launching the first trainings I see problems when importing unsloth

```bash
source ~/arc25_env/bin/activate
python -c "import unsloth"
2025-09-30 06:14:27,085 - datasets - INFO - <module> - PyTorch version 2.7.1 available.
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
INFO 09-30 06:14:29 [__init__.py:241] Automatically detected platform cuda.
🦥 Unsloth Zoo will now patch everything to make training faster!
Traceback (most recent call last):
  File "/root/arc25/scripts/search_and_learn_with_unsloth.py", line 16, in <module>
    from unsloth import FastLanguageModel
  File "/root/arc25_env/lib/python3.12/site-packages/unsloth/__init__.py", line 247, in <module>
    from .models import *
  File "/root/arc25_env/lib/python3.12/site-packages/unsloth/models/__init__.py", line 15, in <module>
    from .llama     import FastLlamaModel
  File "/root/arc25_env/lib/python3.12/site-packages/unsloth/models/llama.py", line 52, in <module>
    from .vision import FastBaseModel
  File "/root/arc25_env/lib/python3.12/site-packages/unsloth/models/vision.py", line 87, in <module>
    from unsloth_zoo.vllm_utils import (
  File "/root/arc25_env/lib/python3.12/site-packages/unsloth_zoo/vllm_utils.py", line 63, in <module>
    from unsloth.models.vision import VLLM_SUPPORTED_VLM
ImportError: cannot import name 'VLLM_SUPPORTED_VLM' from partially initialized module 'unsloth.models.vision' (most likely due to a circular import) (/root/arc25_env/lib/python3.12/site-packages/unsloth/models/vision.py)

# try to reinstall unsloth, but does not solve the problem
pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth==2025.9.3 unsloth_zoo

# install python 3.10
apt update
apt install -y software-properties-common
add-apt-repository ppa:deadsnakes/ppa -y
apt update
apt install -y python3.10 python3.10-venv python3.10-dev
/usr/bin/python3.10 -m venv ~/arc25_env310
source ~/arc25_env310/bin/activate
python -V        # should print 3.10.x
pip install -U pip
# Does not solve the problem either
# This is the solution
pip install unsloth_zoo==2025.9.6

```

#### Doubts and suggestions

- When I start a container and select some type of machine. Do I pay for the machine when installing python or other things? Should I select a cheap machine for development and a expensive one for training? Yes, we are charged for using the workstation. So is better to use a cheap workstation to launch jobs.
- I lost internet when using wireguard, solved.
- I have created a dataset but does not seem to be working
- Should I stop the container after creating the environment? Yes
- When trying to train on "canada-a100-x1-p3-ws" I get `Failed to build shape list from priority list, as no shapes matched the priority list`, same for "canada-h100-x1-p3-ws". Why some machines are only available for workstation and not for burst experiments?
- It would be nice to be able to sort the shapes by cost, and show also the cost per GPU (not just the total cost)
- It would be nice to be able to batch delete previous experiments, to have a cleaner interface in the web

#### Training on the Sydney Cluster

I have noticed that they charge $1.25 hour per GPU on the Sydney Cluster, and slightly less for
having a container active. On a machine with 4 GPUs that would be around $1.5 per 3090 GPU. Thus
it does not have too much sense to use them considering that they charge $2.10 for an H100 GPU.
So the rest of this section does not have too much sense, but I leave it for reference.

I haven't received a reply of how to train on the Sydney Cluster. Thus I have had the idea to use
workstations for training.

The idea is to request workstations with a few GPUs, and launch search and learn experiments there.
I won't be saving anything so in theory I should be capable of doing it.

I can create workstations on the Sydney Cluster with 4 GPUs, and I have to attach the BARC model to them.
Running htop shows that the machines have 32 cores and 378GB of RAM, more than enough.

```bash
# copy these two files from the first workstation
vim .ssh/id_ed25519
chmod 600 .ssh/id_ed25519
vim secrets.sh #export WANDB_API_KEY=
chmod +x secrets.sh
# now install all the dependencies
apt update && apt install -y python3-dev python3-pip python3-virtualenv git nano vim htop screen nvtop
git config --global user.name "Guillermo Barbadillo"
git config --global user.email "guillermobarbadillo@gmail.com"
git clone git@github.com:ironbar/arc25.git
python3 -m virtualenv ~/arc25_env
source ~/arc25_env/bin/activate
pip install -r arc25/requirements.txt
pip install unsloth_zoo==2025.9.6 # I should update the requirements
# skip this step by now because it is very slow
#MAX_JOBS=2 python -m pip install flash-attn==2.6.3 --no-build-isolation
```

## Results

## Conclusion

## Next steps

## TODO

- [ ] Strong compute
  - [x] Clone arc25 repo, for that I have to add a new public key to github.
  - [x] Create python environment
  - [x] Create dataset for BARC induction model
  - [x] Add data to repo for simplicity
  - [x] Create a sample RL training script
  - [x] Launch first experiment, with wandb, saving to artifacts
  - [x] Create multiple experiments
  - [x] How to get the artifacts? It seems I can make the artifacts available to a running workstation
  - [ ] Awaiting for answers to my doubts
  - [ ] Use workstations for training
    - I'm really charged? 1,292.63 at the start of the experiment
    - $1,292.42 while the experiment is running, some charge has already happened. I see $1.32 on pending charges.
    - $1,287.19 after stopping the experiment. 
    - So yes, I'm being charged for using the Sydney Cluster


