# water-level-prediction
Hackathon AI.BAY2023 Challenge
link: https://taybone2305.github.io/water-level-prediction-passau/

# prediction

*note*: predictions are done for station 12

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

