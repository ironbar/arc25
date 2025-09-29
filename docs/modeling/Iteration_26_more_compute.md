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

#### Doubts

- When I start a container and select some type of machine. Do I pay for the machine when installing python or other things? Should I select a cheap machine for development and a expensive one for training?

## Results

## Conclusion

## Next steps

## TODO

- [ ] Strong compute
  - [ ] Clone arc25 repo
  - [ ] Create python environment
  - [ ] Create dataset for BARC induction model
  - [ ] Add data to repo for simplicity
  - [ ] Launch first experiment, with wandb, saving to artifacts
  - [ ] Create multiple experiments
