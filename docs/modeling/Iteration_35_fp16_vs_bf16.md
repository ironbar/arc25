# Iteration 35. FP16 vs BF16

_01-11-2025_

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.

<details>
  <summary>Click to expand/collapse this section</summary>
</details>
--->

## Goal

Can we prevent RL training collapse if we use FP16 instead of BF16?

## Motivation

Yesterday I saw a lof of people talking in Twitter about using FP16 instead of BF16 for RL training.

![alt text](res/1761974415868_image.png)

They were refearing to the paper [Defeating the Training-Inference Mismatch via FP16](https://arxiv.org/abs/2510.26788).

> Reinforcement learning (RL) fine-tuning of large language models (LLMs) often suffers from instability due to the numerical mismatch between the training and inference policies. While prior work has attempted to mitigate this issue through algorithmic corrections or engineering alignments, we show that its root cause lies in the floating point precision itself. The widely adopted BF16, despite its large dynamic range, introduces large rounding errors that breaks the consistency between training and inference. In this work, we demonstrate that simply reverting to **FP16** effectively eliminates this mismatch. The change is simple, fully supported by modern frameworks with only a few lines of code change, and requires no modification to the model architecture or learning algorithm. Our results suggest that using FP16 uniformly yields more stable optimization, faster convergence, and stronger performance across diverse tasks, algorithms and frameworks. We hope these findings motivate a broader reconsideration of precision trade-offs in RL fine-tuning.

The challenge is almost over, but I haven't been able to find the root of the RL training collapse. So it
would be nice to see that doing this simple change solves the problem.

## Development

The idea is to find a previous training configuration that collapsed quickly (ideally in less than 24h) and run that configuration with bf16 and fp16 and hopefully see that one training collapses and the other does not.

### Cluster experiments

Remove the repetition penalty, and set beta to its default value.

<details>
  <summary>Click to expand/collapse this section</summary>

```bash
export DTYPE=bfloat16
export FOLDER=2025-11-01-rl-fp16
export BETA=0.001
export REPETITION_PENALTY=1.00
export LEARNING_RATE=4e-6
export NUM_GENERATIONS=8
export ACUM_STEPS=1
export N_CPUS=20
export LORA_R=1
export EPOCHS=1
export REWARD_NAME=arc-v2-no-pixel-score
export EXPERIMENT_NAME=${DTYPE}_${LORA_R}lora_lr${LEARNING_RATE}_${REWARD_NAME}_epochs${EPOCHS}_${NUM_GENERATIONS}gen_${ACUM_STEPS}accum-steps_repetition-penalty-${REPETITION_PENALTY}_masked-truncate_unquantized_beta${BETA}
condor_submit train.condor command="
python /mnt/scratch/users/gbarbadillo/arc25/arc25/scripts/rl_code_finetuning.py \
--dtype ${DTYPE} \
--beta ${BETA} \
--no-load-in-4bit \
--reward-name ${REWARD_NAME} \
--num-generations ${NUM_GENERATIONS} \
--gradient-accumulation-steps ${ACUM_STEPS} \
--learning-rate ${LEARNING_RATE} \
--lora_r ${LORA_R} \
--repetition-penalty ${REPETITION_PENALTY} \
--epochs ${EPOCHS} \
--mask-truncated-completions \
--scale-rewards batch \
--gpu_memory_utilization 0.3 \
--warmup-ratio 0.001 \
--max-seq-length 9700 \
--max-completion-length 1024 \
--n-jobs ${N_CPUS} \
--model-path /mnt/scratch/users/gbarbadillo/arc25/models/Llama-3.1-ARC-Potpourri-Induction-8B \
--dataset-path /mnt/scratch/users/gbarbadillo/arc25/data/barc/dataset_100k.json.gz \
--output-dir /mnt/scratch/users/gbarbadillo/arc25/trainings/${FOLDER}/${EXPERIMENT_NAME}" -append request_gpus=1 -append request_cpus=${N_CPUS} -append request_memory=128G --append 'requirements = (TARGET.Machine == "calculon21.das-nano.com")'
```

</details>

## Results

## Conclusion

## Next steps

## TODO

- [ ] Run a fp16 vs bf16 training
- [ ] Do I also have to look the precision used for VLLM?
