# Iteration 33. RL with BARC data

_14-10-2025_

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.

<details>
  <summary>Click to expand/collapse this section</summary>
</details>
--->

## Goal

Train with RL from BARC synthetic datasets. Does it solve the collapsing problems? Does it produce stronger models?

## Motivation

Maybe the collapsing problems comes from repeating the 400 training tasks over and over. Since we
have the BARC synthetic datasets readily available, we should try to use them to see if we can
get stronger models and solve the training collapse problem.

## Development

The Heavy version of the [dataset](https://huggingface.co/datasets/barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems)
is the most interesting one because it uses more seed functions and uses the stronger gpt4 model for description generation.

### Verify that I can train locally

```bash
python scripts/rl_code_finetuning.py \
--dataset-path /mnt/hdd0/Kaggle/arc25/data/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems/dataset_100k.json.gz \
--epochs 1 \
--output-dir /mnt/hdd0/Kaggle/arc25/trainings/2025-10-14-debug-BARC/debug
```

Works perfectly.

## Results

## Conclusion

## Next steps

## TODO

- [ ] Download and curate the synthetic datasets: https://huggingface.co/datasets/barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems
