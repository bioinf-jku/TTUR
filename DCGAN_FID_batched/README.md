# DCGAN for CelebA evaluated with FID (batched version)

DCGAN fork from https://github.com/carpedm20/DCGAN-tensorflow

Precalculated real world / trainng data statistics can be downloaded from:
http://bioinf.jku.at/research/ttur/ttur.html

## Usage
- Download the precalculated statistics (see above) and save them into the "stats" folder.
- Add model path in file run.sh at line 30
- Add data path in file run.sh at line 31
- run the command: bash run.sh <name_of_run> <disc. lr> <gen. lr> main.py

## FID evaluation: parameters fid_n_samples and fid_sample_batchsize
The evaluation of the FID needs the comparison between precalculated statistics of real world data vs statistics of generated data.
The calculation of the latter is a tradeoff between number of samples (the more the better) and available hardware. Two parameters
in run.sh are concerned with this calculation: fid_n_samples and fid_sample_batchsize. The first parameter specifies the number of
generated samples on which the statistics are calculated. Since this number should be high, it is very likely that it is not possible 
to generate this amount of samples at once. Thus the generation process is batched with batches of size fid_sample_batchsize. 
