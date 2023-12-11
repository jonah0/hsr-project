import pathlib
import pandas as pd


cwd = pathlib.Path(__file__).parent.resolve()


def sum_db1b_passengers(year: int, quarter: int) -> pd.DataFrame:
    print(f'sum_db1b_passengers({year}, {quarter})', '...')
    if year is None or quarter is None:
        raise Exception('Must specify both year and quarter!')

    db1b = pd.read_csv(cwd.joinpath(
        f'./data/db1b-survey/market/Origin_and_Destination_Survey_DB1BMarket_{year}_{quarter}.csv'
    ))

    cols_of_interest = [
        "Year",
        "Quarter",
        "Origin",
        "Dest",
        "NonStopMiles",
        "Passengers",
    ]

    groupby_cols = [
        "Year",
        "Quarter",
        "Origin",
        "Dest",
        "NonStopMiles",
    ]

    df = db1b[cols_of_interest].groupby(by=groupby_cols).sum().reset_index()
    # NOTE: DB1B is a 10% survey, so multiply passenger numbers by 10 to get true number
    df['Passengers'] *= 10
    return df


year = 2022
quarter_dfs = []
for qt in range(1, 5):
    df = sum_db1b_passengers(year=year, quarter=qt)
    quarter_dfs.append(df)
    outfilepath = f'./data/db1b-survey/market/processed/db1b-passenger-vol-{year}-{qt}.csv'
    df.to_csv(cwd.joinpath(outfilepath), index=False)

# concatenate all 'quarter' dataframes together
df_annum = pd.concat(quarter_dfs)
groupby_cols = [
    "Year",
    "Origin",
    "Dest",
    "NonStopMiles",
]

# drop 'Quarter' column, group by route (city pair), and sum 'Passengers' column
df_annum = df_annum.drop(columns='Quarter').groupby(groupby_cols).sum(
).reset_index().sort_values(by=['Origin', 'Dest', 'Passengers'])

# write to csv
outfilepath = f'./data/db1b-survey/market/processed/db1b-passenger-vol-{year}-full.csv'
df_annum.to_csv(cwd.joinpath(outfilepath), index=False)
print(df_annum)
print('Total annual passengers:', df_annum['Passengers'].sum())
