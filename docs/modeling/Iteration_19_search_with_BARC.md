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
