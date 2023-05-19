# experiments_emolit
Train a model like in the emotions in literature paper.

Data: https://zenodo.org/record/7883954

Extract the data into the `data` dir so it looks like this: `data/emolit`.

Install the requirements in `requirements.txt` (consider virtualenv).

Change any parameters in the `soft_train.py` file (e.g. encoder model, batch
size, number of epochs, ...).

Run: `python soft_train.py`. This should train a model and save it to the `model` directory.
