# Social Emotion Classification

This repository contains all the work done on Social Emotion Classification on 
[Sharing emotions at scale: The Vent dataset, N. Lykousas et 
al.](https://arxiv.org/pdf/1901.04856.pdf)

# Running the Experiments

We provide a single python script meant to execute all the experiments described in the paper. 
Use:

```bash
$ python src/run.py
```

To run all experiments. The program will detect if you are in a Slurm-enabled cluster like the
one we use at Universitat Pompeu Fabra, or start as many threads as it can find. You will need 
a setup that has GPUs lest you want to die of old age!

# Contact

Nurudin Álvarez González `nalvarez [AT] ntent [DOT] com`.

