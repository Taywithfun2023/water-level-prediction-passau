set -ex

if [[ ! -e venv ]]; then
    python -m virtualenv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

if [[ ! -e dev/test-data.csv ]]; then
    echo "*** PREPARING TEST DATA"
    python src/data_prep.py test/test-data dev/test-data.csv
fi

echo "*** PREDICTING ON TEST DATA"
python src/predict.py dev/model-o1-tNone.pkl dev/model-o1-t700.pkl dev/test-data.csv dev/test-predictions.json

