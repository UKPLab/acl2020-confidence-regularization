# Mind the Trade-off: Debiasing NLU Models without Degrading the In-distribution Performance

### Code will be released soon!

The repository contains the code to reproduce our work in debiasing NLU models without in-distribution degradation


```
@InProceedings{UtamaDebias2020,
  author    = {Utama, P. Ajie and Moosavi, Nafise Sadat and Gurevych, Iryna},
  title     = {Mind the Trade-off: Debiasing NLU Models without Degrading the In-distribution Performance},
  booktitle = {Proceedings of the 58th Conference of the Association for Computational Linguistics},
  month     = jul,
  year      = {2020},
  publisher = {Association for Computational Linguistics}
}
```

> **Abstract:** Models for natural language understanding (NLU) tasks often rely on the idiosyncratic biases of the dataset, which make them brittle against test cases outside the training distribution. 
Recently, several proposed debiasing methods are shown to be very effective in improving out-of-distribution performance. However, their improvements come at the expense of performance drop when models are evaluated on the in-distribution data, which contain examples with higher diversity. 
This seemingly inevitable trade-off may not tell us much about the changes in the reasoning and understanding capabilities of the resulting models on broader types of examples beyond the small subset represented in the out-of-distribution data.
In this paper, we address this trade-off by introducing a novel debiasing method, called confidence 
regularization, which discourage models from exploiting biases while enabling them to receive enough incentive to learn from all the training examples. We evaluate our method on three NLU tasks and show that, in contrast to its predecessors, it improves the performance on out-of-distribution datasets (e.g., 7pp gain on HANS dataset) while maintaining the original in-distribution accuracy.


## Data


## Running the experiments
**(change this as needed!)**

```
$cd bla/bla/bla
$some_cool_commands_here
```


Contact person: Ajie Utama, utama@ukp.informatik.tu-darmstadt.de
https://www.ukp.tu-darmstadt.de/
https://www.tu-darmstadt.de/

Please reach out to us for further questions or if you encounter any issue.