# Hackathon AI.BAY2023 Challenge
Passau, surrounded by the Inn, Danube, and Ilz rivers, is at high risk of flooding due to its hilly terrain and densely populated river banks. To mitigate this risk, a comprehensive solution has been created and made available on a website that utilises a predictive computational pipeline.

Team members: Emilio Dorigatti, Akshita Agarwal, Yeva Suzkom Karim Belaid, Xenofon Giannoulis, Tayfun Bönsch.

# Website

# Predictions

**note**: predictions are done for station 12

## Training and Testing
 - training data:
    - precipitation: `data/precipitation/data/data_*.csv`
    - water level: `data/water_level/messtation-*.csv`
 - run `train.sh` to train the models (will be saved in `dev/`)
 - testing data:
    - precipitation: `test/test-data/precipitation/data/data_*.csv`
    - water level: `test/test-data/water_level/messtation-*.csv`
    - must be in the **same** format (file names and content) as the training data
 - run `test.sh` to obtain predictions
    - metrics will be printed to stdout and predictions saved in `dev/`

## Explanation
The predictions are based on the water level measured at each station and the sum of precipitations in the last three days.
In order to exclude outliers, the predictions are based on the *third largest* water level measured in the next 1, 2, 3, 4 and 5 days.
We trained regression models to directly predict the (third largest) water level, as well as classification models to predict the probability that the (third largest) water level is above 600, 700, 740, and 770 centimeters.
The thresholds, especially the latter two, were chosen based on [the official guidelines](https://www.hochwasser-passau.de/en/#pegelstaende), and the resulting probabilities can be used in aggregate to derive risk scores.

We used [TPOT](https://github.com/EpistasisLab/tpot) to find the best model and its hyperparameters on the regression task and used the analogous version for classification, using data from 1997 to 2019 for the optimization and the data for 2020 as held-out for evaluation and the final website demonstration.
The chosen models were linear support vector machines for regression and classification, the latter calibrated via isotonic regression on a 5-fold time-splitted cross-validation.
The training script computes absoluted and squared error for regression and average precision score and accuracy for classification for each threshold and prediction horizon.

## Web Application
The website displays the model's prediction for 2020 and is accessible at [https://taybone2305.github.io/water-level-prediction-passau](https://taybone2305.github.io/water-level-prediction-passau).
It is hosted on github pages and works both on desktop and mobile.

Structure:
  1) Navigation bar: jump to sections & search function
  2) Map: overview of Passau (caption); future feature: show (non-)accessible areas with colour patches
  3) Date: choosing a date is necessary in order to get the desired results
  4) Data: analysis based on the used model (see upper section of README) -> alerts (pictures), predictions, historical data
  5) Contact: important information regarding contact persons for citizen
  6) Social Media: footer
