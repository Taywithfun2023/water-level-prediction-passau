set -ex
if [[ ! -e venv ]]; then
    python -m virtualenv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

mkdir -p dev

if [[ ! -e dev/train-data.csv ]]; then
    echo "*** PREPARING TRAINING DATA"
    python src/data_prep.py data dev/train-data.csv
fi

echo "*** TRAINING MODELS"
python src/train.py dev/train-data.csv dev/

