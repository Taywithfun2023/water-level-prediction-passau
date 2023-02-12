import pandas as pd
import click
import os


def read_precipitation_data(prep_dir):
    dfs = []
    for fname in sorted(os.listdir(prep_dir)):
        if fname.startswith('data_') and fname.endswith('.csv'):
            print('reading', prep_dir + fname)
            df = pd.read_csv(prep_dir + fname)
            df['station'] = fname.split('.')[0].split('_')[1]
            dfs.append(df)
    df = pd.concat(dfs)
    df['date'] = pd.to_datetime(df['Zeitstempel'])
    df = df.pivot(index='date', columns='station', values='Wert')
    df.columns = [f'prec-{c}' for c in df.columns]
    return df


def read_water_level(levls_dir):
    dfs = []
    for fname in sorted(os.listdir(levls_dir)):
        if fname.startswith('messtation-') and fname.endswith('.csv'):
            print('reading', levls_dir + fname)
            df = pd.read_csv(levls_dir + fname)
            df['date'] = pd.to_datetime(df['Datum'])
            df['level'] = df['Wasserstand [cm]'].str.split(',').str[0]
            df = df[~df['level'].isna()]
            df['level'] = df['level'].astype(int)
            q = df.resample('D', on='date')['level'].agg([
                # get third largest observation (idea: max could be an outlier)
                # there are 96 observations per day
                lambda x: sorted(x)[-3] if len(x) > 0 else float('nan')
            ]).dropna()
            q.columns = ['max3']
            q['station'] = fname.split('.')[0].split('-')[1]
            dfs.append(q)
    df = pd.concat(dfs)
    df = df.pivot_table(index='date', columns='station', values='max3')
    df.columns = [f'level-{c}' for c in df.columns]
    return df


def read_data(base_dir):
    df_precip = read_precipitation_data(f"{base_dir}/precipitation/data/")
    df_water = read_water_level(f'{base_dir}/water_level/')
    df = pd.merge(df_precip, df_water, left_index=True, right_index=True)
    df = df.reset_index()
    return df


@click.command()
@click.argument('data-dir')
@click.argument('output')
def main(data_dir, output):
    df = read_data(data_dir)
    df.to_csv(output, index=False)


if __name__ == '__main__':
    main()
