# CoolEffNet

CoolEffNet is a set of tools written in Python for predicting film cooling effectiveness
in flat plate experiments, using a technique called
[Bayesian Neural Network Ensembles](https://doi.org/10.48550/arXiv.1811.12188).
A set of pre-trained ensembles are included along with a tool to train network ensembles on the user's own data and save them.
The mean prediction of the data has 19-36% less root-mean-squared error compared to
using Colban's and Baldauf's correlations over the same dataset.
A technical report detailing the model and its advantages can be found here.

## Installation

Installation is via `pip`.
Using your favourite shell, enter: 

`pip install cooleffnet`

## Usage

`cooleffnet` is the predictor tool, and `cooleffnet-train` is capable of training ensembles.
Both have large variety of options and an extensive help accessible via the `-h` and `--help` options.

### Options

`cooleffnet`



### Examples


#### From the command-line


#### From Python
