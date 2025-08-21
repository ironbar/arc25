# Iteration 19. Search with BARC

_18/08/2025_

## Goal

Use the induction model from Boostrapping ARC (BARC) to search code solutions. This will stablish
a baseline score that could be later be used to explore methods to improve the efficiency of the search.

## Motivation

I believe I can validate my ideas using the model from BARC. That saves me the process of generating
a DSL and training a model on it. Later if I validate my ideas and find that the DSL is incomplete, I
could devote to the task of creating the ultimate DSL. But first I have to validate that we can
improve dramatically the accuracy of an induction model by learning from the search results.

## Development

Links:

- https://github.com/xu3kev/BARC/tree/master
- https://huggingface.co/barc0/Llama-3.1-ARC-Potpourri-Induction-8B

### Grid encoder

I have developed a new grid encoder that represents the grids with colors with their names:

```
Black Blue Red Green Yellow Gray Pink Orange Purple Brown
```

After being tokenized this does not use more tokens than my implementation, the spaces are encoded with the words.

However I have noticed that the first words of each row are represented differently by the encoder.
Probably this is not the best option but the model was trained this way. Below you can see how the
tokenizer tokenizes a 2x2 matrix with zeros.

```
['Black', 'ĠBlack', 'Ċ', 'Black', 'ĠBlack']
```

## Results

### Comparison of pass@n in the different datasets

The plot below shows the pass@n for the different datasets versus the number of predictions. I believe
this validates my current implementation because I get similar or better numbers to the ones reported
in the BARC paper. This could be happening due to using a higher temperature and/or using more input samples.

![alt text](res/1755754930851_image.png)

- 2024 datasets do not show signs of stopping if we increase the number of predictions, however the
  2025 dataset does not improve when increasing the number of predictions from 256 to 1024. This might
  be a sign that the 2025 dataset is much harder than the previous one.
- Interesting to see that the training set is not easily solved, this might indicate that there is room
  for improvement

### Other metrics analysis

| dataset         | n_preds | valid code | valid outputs | unique outputs | pixel similarity | correct grids | pass_rate | pass@n |
|-----------------|---------|------------|---------------|----------------|------------------|---------------|-----------|--------|
| training-2024   | 248     | 100.0%     | 82.0%         | 38.1%          | 61.9%            | 15.0%         | 12.40%    | 58.00% |
| evaluation-2024 | 568     | 100.0%     | 75.9%         | 40.9%          | 57.1%            | 3.0%          | 1.96%     | 21.00% |
| evaluation-2025 | 1560    | 100.0%     | 72.8%         | 39.9%          | 50.3%            | 0.1%          | 0.051%    | 1.67%  |

- The ratio of unique outputs is quite good, when [searching with base models](./Iteration_16_search_with_base_models.md#effect-of-the-number-of-predictions-with-independent-search)
I would only get around 40% unique outputs when doing 16 predictions, and the rate lowered to 20% when
doing 512 predictions.
- Pixel similarity might not be a good metric, the difference between the datasets is small, but the differences in pass rate are huge.
  This could be caused by the binary nature of the arc tasks, they are either correct or wrong.

### Optimizing the number of predictions for throughput



## Conclusion

## Next steps

- Induction results from the BARC paper are obtained with 20k samples, that gets 38% on the validation set.
  With the hardware available at Kaggle I could make around 2k predictions per task. That would yield
  a score around 20% on the validation set. 500 predictions yields around 15%, 200 around 10%, not sure how many are needed to get 5% score.
  So maybe my approach from last year was not that bad, I was simply not making that many predictions.
  Maybe reinforcement learning could increase the efficiency of the model at test-time, reducing the number
  of required predictions.
![alt text](res/1755582102516_image.png)

## TODO

- [ ]
