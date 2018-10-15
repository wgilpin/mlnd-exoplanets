# Detecting Exoplanets from NASA Kepler Data  

**Machine Learning Engineer Nanodegree**  
**Capstone Proposal**  
Will Gilpin  
July 25, 2018  


## Readme.md

### Setup
The included Jupyter Notebook was run under a Google Cloud click-to-deploy instance of the `Deep Learning VM`.

This instance included CUDA support for the 2 GPUS in use. If you are not using GPUs you will need to replace the `CuDNNLSTM` calls with basic LSTM calls.

### Libraries
The code runs on python 2.7, and requires the following libraries, in no particular order
```
tensorflow
keras
numpy
scipy
scikit-learn
matplotlib
```

### Data

Data files required to run the code are the `exoTest` and `exoTrain` labelled time-series datasets from https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data.

These are obtainable from the CLI as follows
```
kaggle datasets download -d keplersmachines/kepler-labelled-time-series-data
```

or from the following links:

[exotest.csv](https://www.dropbox.com/s/4migonsgpp50xma/exoTest.csv?dl=0)

[exoTrain.csv](https://www.dropbox.com/s/pbv7qw35d5o5dka/exoTrain.csv?dl=0)
