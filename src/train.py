import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
import click
import json
import os
import io
from collections import Counter
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from tqdm import trange
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, LinearSVC
import joblib



# https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html
class NaivelyCalibratedLinearSVC(LinearSVC):
    def fit(self, X, y):
        super().fit(X, y)
        df = self.decision_function(X)
        self.df_min_ = df.min()
        self.df_max_ = df.max()

    def predict_proba(self, X):
        df = self.decision_function(X)
        calibrated_df = (df - self.df_min_) / (self.df_max_ - self.df_min_)
        proba_pos_class = np.clip(calibrated_df, 0, 1)
        proba_neg_class = 1 - proba_pos_class
        proba = np.c_[proba_neg_class, proba_pos_class]
        return proba


def make_data(df, offsets):
    train_data = []
    test_data_by_offset = {o: [] for o in offsets}
    dates = []

    past_days = 3
    for start_day in trange(past_days, len(df) - max(offsets)):
        x = df.iloc[start_day - past_days:start_day]
        train_data.append(
            np.concatenate([
                # all levels
                x[[c for c in df.columns if c != 'date']].values.flatten(),
                # sum of past precip
                x[[c for c in df.columns if 'prec-' in c]].sum(axis=0).values.flatten(),
            ])
        )
        dates.append(df.iloc[start_day]['date'])
        d = dates[-1]
        for o in offsets:
            yy = df.iloc[start_day:start_day+o]['level-12'].max()
            assert np.isfinite(yy)
            test_data_by_offset[o].append(yy)

    train_data = np.array(train_data)
    test_data_by_offset = {k: np.array(v).reshape((-1, 1)) for k, v in test_data_by_offset.items()}

    return train_data, test_data_by_offset, dates


@click.command()
@click.argument('train-data')
@click.argument('output-folder')
def main(train_data, output_folder):
    offsets = [1, 2, 3, 4, 5]

    print('reading data')
    df = pd.read_csv(train_data).dropna()
    df['date'] = pd.to_datetime(df['date'])
    train_data, test_data_by_offset, dates = make_data(df, offsets)
    train_test_cut = -359
    thresholds = [600, 700, 740, 770, None]

    pred_df = []
    for t in thresholds:
        for o in offsets:
            # data prep
            X_train, y_train = train_data[:train_test_cut], test_data_by_offset[o][:train_test_cut]
            X_test, y_test = train_data[train_test_cut:], test_data_by_offset[o][train_test_cut:]
            xs = StandardScaler().fit(X_train)

            ws = y_train.flatten()
            ws = ws - ws.min()
            ws = ws / ws.max()

            if t is None:
                # predict for regression
                sv = LinearSVR(C=25.0, dual=False, epsilon=0.001, loss="squared_epsilon_insensitive", tol=1e-05)
                sv.fit(xs.transform(X_train), y_train.ravel(), sample_weight=np.exp(ws))
                ps = sv.predict(xs.transform(X_test))

                print('offset', o)
                print('  abs error', np.mean(np.abs(ps - y_test.ravel())))
                print('  sq error', np.mean((ps - y_test.ravel())**2))

            else:
                # predict for classification
                y_train = (y_train.ravel() > t).astype(int)
                y_test = (y_test.ravel() > t).astype(int)
                if not any(y_test):
                    continue

                sv = CalibratedClassifierCV(
                    NaivelyCalibratedLinearSVC(C=25.0, dual=False, loss="squared_hinge", tol=1e-05),
                    cv=TimeSeriesSplit(5), method="isotonic"
                )
                sv.fit(xs.transform(X_train), y_train)
                ps = sv.predict_proba(xs.transform(X_test))[:, 1]

                print('offset', o, 'threshold', t)
                print('  aps:', average_precision_score(y_test.flatten(), ps))
                print('  acc:', np.mean(((y_test.flatten() > 0.5) == (ps > 0.5))))

            pred_df.append(pd.DataFrame({
                'date': dates[train_test_cut:],
                'prediction': ps,
                'observed': y_test.flatten(),
                'offset': o,
                'threshold': t,
            }))

            joblib.dump({
                'model': sv,
                'scaler': xs,
                'threshold': t,
            }, f'{output_folder}/model-o{o}-t{t}.pkl')

    pdf = pd.concat(pred_df)
    pdf['date'] = pdf['date'].astype(str)

    rows = [dict(zip(pdf.columns, c)) for c in pdf.itertuples(False, None)]
    with open(f'{output_folder}/train-predictions.json', 'w') as f:
        json.dump(rows, f)


if __name__ == '__main__':
    main()
