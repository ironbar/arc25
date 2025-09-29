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
cd ~
git clone git@github.com:ironbar/arc25.git
python3 -m virtualenv ~/arc25_env
source ~/arc25_env/bin/activate
pip install -r arc25/requirements.txt
pip install flash-attn==2.6.3 --no-build-isolation
```

- Created dataset from huggingface [barc0/Llama-3.1-ARC-Potpourri-Induction-8B](https://huggingface.co/barc0/Llama-3.1-ARC-Potpourri-Induction-8B)
- After installing the requirements stopping the machine took more time, probably due to saving the environment

#### Doubts

- When I start a container and select some type of machine. Do I pay for the machine when installing python or other things? Should I select a cheap machine for development and a expensive one for training?
- I lost internet when using wireguard, solved.
- I have created a dataset but does not seem to be working

## Results

## Conclusion

## Next steps

## TODO

- [ ] Strong compute
  - [x] Clone arc25 repo, for that I have to add a new public key to github.
  - [ ] Create python environment
  - [ ] Create dataset for BARC induction model
  - [ ] Add data to repo for simplicity
  - [ ] Launch first experiment, with wandb, saving to artifacts
  - [ ] Create multiple experiments
