## Introduction

[fasterText](https://github.com/eli-osherovich/fasterText) is a library designed for fast and accurate word embeddings. While originated from [fastText](https://fasttext.cc/) it may not be fully compatible with it.


## Requirements
As a pre-requisite you will need:
* [Intel Integrated Performance Primitives](https://software.intel.com/en-us/intel-ipp)
* [Boost tokenizer](https://www.boost.org/doc/libs/1_66_0/libs/tokenizer/)
* a modern C++ compiler (with good C++11 support)

## Building fasterText

```
$ wget https://github.com/eli-osherovich/fasterText/archive/master.zip
$ unzip master.zip
$ cd fasterText-master
$ make
```

This will produce object files for all the classes as well as the main binary `fastertext`.
If you do not plan on using the default system-wide compiler, update the two macros defined at the beginning of the Makefile (CC and INCLUDES).

## Word representation learning

In order to learn word vectors, do:

```
$ ./fastertext skipgram -input data.txt -output model
```

where `data.txt` is a training file containing `UTF-8` encoded text.
By default the word vectors will take into account character n-grams from 3 to 6 characters.
At the end of optimization the program will save two files: `model.bin` and `model.vec`.
`model.vec` is a text file containing the word vectors, one per line.
`model.bin` is a binary file containing the parameters of the model along with the dictionary and all hyper parameters.
The binary file can be used later to compute word vectors or to restart the optimization.

## Comparison with fastText

Following are the results that were obtained (*) with latest versions of fastText and fasterText on a full dump of the English Wikipedia (4.5B words) on two popular benchmarks: WS353 and RW.

​Framework | Time (mins) | Embedding Dimensionality | Epochs | ​WS353 (OOV%) | ​RW (OOV%) | ​RW common (OOV%)
----------|-------------|--------------------------|--------|--------------|-----------|----------------------
​FastText  |     ​122     ​| 100       ​               | 5  ​    | 71 (0%)      | ​44 (5%)   | ​44 (3%)
​FasterText|     ​62      ​| 100       ​               | 5      | ​72 (0%)      | ​43 (3%)   | ​43 (3%)
​FastText  |	​813     ​| 300                      | ​10     | ​74 (0%)      | ​49 (5%)   | 49 (3%)
​FasterText|     ​255     ​| 300   ​                   | 10     | ​74 (0%)      | ​47 (3%)   | ​48 (3%)


(*) The results above were obtained on an Amazon EC2 instance m5.x24 using 96 threads.
