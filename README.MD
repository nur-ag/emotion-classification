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

# Running the models

Beyond reproducing the results, we invite researchers to access, reuse, and evaluate the best models trained on the GoEmotions and Vent data sets. We provide Docker images containing the models to be called as convenient REST APIs through [DockerHub](https://hub.docker.com/r/nursystems/emotion-models). Please refer to `src/serve.py` for details on the interface.

# Contact

Nurudin Álvarez González `msg [AT] nur [DOT] wtf`.

