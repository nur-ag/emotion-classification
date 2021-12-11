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

## Running models locally

Since using Docker might introduce unwanted overhead for researchers that simply want to explore models, we also provide lightweight access to the two models in the Emotion UI. Simply create a `release` folder in the project root and download the models:

```bash
mkdir release
wget https://emotion-classification-models.s3.eu-west-1.amazonaws.com/GoEmotions.pkl -O release/
wget https://emotion-classification-models.s3.eu-west-1.amazonaws.com/Vent.pkl -O release/
```

You will then be able to use the same `src/serve.py` script to produce emotion predictions locally. Please note that you will need to install all dependencies listed in `requirements-release.txt` in order to mimick the execution environment in the docker images (see `Dockerfile` for details). The script below shows how to obtain model predictions from the best-performing model trained on the Vent data set:

```python
import sys
import asyncio

sys.path.append('src/')

from src import serve as S

request = S.PredictionRequest(text='a sad input string for a bad day using vent', model='Vent')
result = asyncio.run(S.predict(request))

# The result is a dictionary containing 'text', 'model', and 'emotions' (with the results)
# The result['emotions'] is a list of emotions, sorted by probability (e.g. the output of the model)
top_emotion = result['emotions'][0]

# Prints: {'id': 70, 'emotion': 'Sad', 'score': 0.15725864470005035, 'threshold': 0.10463140904903412, 'active': True, 'category': 'Sadness', 'color': '#4682B4'}
print(top_emotion)
```

# Contact

Nurudin Álvarez González `msg [AT] nur [DOT] wtf`.

