# DCGAN for CelebA evaluated with FID (batched version)

DCGAN fork from https://github.com/carpedm20/DCGAN-tensorflow

Precalculated real world / trainng data statistics can be downloaded from:
http://bioinf.jku.at/research/ttur/ttur.html
Be sure to use the batched version.

## Usage
- Download the precalculated statistics (see above) and save them into the "stats" folder.
- Add model path in file run.sh at line 28
- Add data path in file run.sh at line 29
- run the command: bash run.sh <name_of_run> <disc. lr> <gen. lr> main.py
