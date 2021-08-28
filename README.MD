# Uncovering the Limits of Text-based Emotion Detection

This repository contains all the work done on Uncovering the Limits of Text-based Emotion Detection on 
[The Vent Dataset -- Sharing emotions at scale: The Vent dataset, N. Lykousas et 
al.](https://arxiv.org/pdf/1901.04856.pdf) and [The GoEmotions Dataset -- GoEmotions: A Dataset of Fine-Grained Emotions](https://arxiv.org/abs/2005.00547)

# Running the Experiments

We provide a single Bash script meant to execute all the experiments described in the paper. 
Use:

```bash
$ ./bin/replicateExperiments.sh
```

To run all experiments. The program will detect if you are in a Slurm-enabled cluster like the
one we use at Universitat Pompeu Fabra, or start as many threads as it can find. You will need 
a setup that has GPUs lest you want to die of old age!

# Contact

Nurudin Álvarez González `msg [AT] nur [DOT] wtf`.

