# CSGM
This repository contains the implementation of "Towards Collaborative Anti-Money Laundering Among Financial Institutions". 

## Prerequisites
Install the required Python libraries using the following command:

```
pip install pandas numpy tqdm scikit-learn
```

## Dataset
#### AMLSim
The AMLSim dataset is generated with [AMLSim](https://github.com/IBM/AMLSim/tree/master) repostitory. We will open-souce the dataset with Google Drive.

#### AMLWorld
The AMLWorld dataset can be found on [Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/data). To use this data with the provided training scripts, you first need to perform a pre-processing step for the downloaded transaction files (e.g. `HI-Small_Trans.csv`, and `HI-Small_Patterns.txt`):
```
python  data_process/ml_extraction.py --data_path /path/to/kaggle-files/HI-Small_Trans.csv  --pattern_path /path/to/kaggle-files/HI-Small_Patterns.txt
```

The output consists of three files, `accounts.csv`, `transactions.csv` and `alert_accounts.csv` stored in the dictionary `./HI-Small_Trans`
- `accounts.csv`: Contains node IDs along with the institution each node belongs to.

- `transactions.csv`: Contains all transaction records.

- `alert_accounts.csv`: Contains money laundering accounts used for evaluation.

## Usage
To run the experiments with `Prob-CSGM`, you can run, e.g.:

```
python aml_prob.py --data_path ./HI-Small_Trans --rows 7 --bands 100

```
To run the experiments with `Sim-CSGM`, you can run, e.g.:

```
python aml_sim.py --data_path ./HI-Small_Trans --rows 7 --bands 100

```

<div align="center">

| Argument       | Description                  |
| -------------- | ---------------------------- |
| `--rows`       | Number of rows of a band     |
| `--bands`      | Number of bands              |

</div>

## Licence
Apache License
Version 2.0, January 2004