# Dry-Bean-Classification

Dry Bean dataset provided by UC Irvine Machine Learning Repository.


Three Machine Learning algorithm:
1. Softmax from scratch
2. Support Vector Machine from scratch
3. Nueral Network by Keras


## Data Summary

- format: Tabular (xlsx)
- Train : Val : Test = 11000 : 1000 : 1611


| Name | Shape | Details |
| --- | --- | --- |
| Raw Shape | (13611,17) | 16 features + 1 class |
| Input Shape | (13611,17) | 16 features + 1 bias |


| Class | Number |
| --- | --- |
| Barbunya | 1322 |
| Horoz | 1928 |
| Bombay | 522 |
| Seker | 2027 |
| Cali | 1630 |
| Sira | 2636 |
| Dermosan | 3546 |


## Results

Loss by CrossEntropy.
Data normalized.


| Model | Test Acc |
| --- | --- |
| Majority Guess | 25.9466 % | 
| SoftMax | 91.8684 % |
| SVM | 90.1303 % |
| NN | 91.8684 % |


### Train Loss and Validation Accuracy Plots
![Train Loss and Validation Accuracy Plots](/plots/TrainValPlot.PNG)

### Raw Data Tabular
![Raw Data](/plots/RawData.PNG)


## Requirements
- Keras == `2.4.3`
- matplotlib == `3.1.1`
- numpy == `1.21.6`
- pandas == `1.1.2`


## Usage
```sh
# main file contains three methods
python3 main.py
```
