# dev_petastorm

Notes
* The dataset writing script is `dev_dev_petastorm_flashmatchdata.py`
* The reading script is `pytorch_reader_test.py`
* The dataset writing script assumes there are two symbolic links to the same file.
  This is to figure out how to partition a dataset.
* We can write a spark dataframe partitioning by the input file.
* Remains to be seen how a dataset distributed on the cluster will work.
