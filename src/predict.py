import click
from sklearn.metrics import average_precision_score
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, LinearSVC
from train import make_data, NaivelyCalibratedLinearSVC


@click.command()
@click.argument('regression-model-path')
@click.argument('classification-model-path')
@click.argument('data-path')
@click.argument('preds-path')
def main(regression_model_path, classification_model_path, data_path, preds_path):
    df = pd.read_csv(data_path).dropna()
    df['date'] = pd.to_datetime(df['date'])
    test_data, labels_by_offset, dates = make_data(df, offsets=[1])

    # regression
    obj = joblib.load(regression_model_path)
    model, scaler, threshold = obj['model'], obj['scaler'], obj['threshold']
    if threshold is not None:
        raise RuntimeError('classification model passed for regression')
    psr = model.predict(scaler.transform(test_data))
    ysr = labels_by_offset[1].ravel()
    print('MAE:', np.mean(np.abs(psr - ysr)))
    print('MSE:', np.mean((psr - ysr)**2))

    # classification
    obj = joblib.load(classification_model_path)
    model, scaler, threshold = obj['model'], obj['scaler'], obj['threshold']
    if threshold is None:
        raise RuntimeError('regression model passed for classification')
    psc = model.predict_proba(scaler.transform(test_data))[:, 1]
    ysc = labels_by_offset[1].ravel() > threshold
    print('APS:', average_precision_score(ysc, psc))
    print('ACC:', np.mean((ysc > 0.5) == (psc > 0.5)))

    with open(preds_path, 'w') as f:
        json.dump({
            'date': [str(d) for d in dates],
            'regression_prediction': psr.tolist(),
            'classification_prediction': psc.tolist(),
            'regression_observed': ysr.tolist(),
            'classification_observed': ysc.tolist(),
        }, f)


if __name__ == '__main__':
    main()
