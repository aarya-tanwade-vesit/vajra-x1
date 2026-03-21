"""
generate_scenarios.py
─────────────────────
Extracts REAL sequences from the AI4I 2020 dataset (sensors_data.csv)
to build 4 scenario-based telemetry CSVs — one per failure mode.

Each scenario contains:
  - 80 rows of healthy baseline  (from real healthy rows)
  - 100 rows escalating toward the failure  (real rows approaching the limit)
  - 20 rows of the actual failure  (real rows where that failure fired)

Output columns match app.py expectations.
"""
import pandas as pd
import numpy as np

COLS_RENAME = {
    'Air temperature [K]':      'Air_temperature',
    'Process temperature [K]':  'Process_temperature',
    'Rotational speed [rpm]':   'Rotational_speed',
    'Torque [Nm]':              'Torque',
    'Tool wear [min]':          'Tool_wear',
}

TYPE_MAP = {'L': 'L (Low)', 'M': 'M (Medium)', 'H': 'H (High)'}

def load_and_prep(path='sensors_data.csv'):
    df = pd.read_csv(path)
    df.rename(columns=COLS_RENAME, inplace=True)
    df['Machine_Type'] = df['Type'].map(TYPE_MAP)
    # Compute OSF proximity for sorting
    df['OSF_proximity'] = df['Tool_wear'] * df['Torque'] / 12000.0
    return df

def build_scenario(df, failure_col, n_healthy=80, n_approach=100, n_fail=20, seed=42):
    """
    Pull real rows from the dataset ordered so they tell a degradation story.
    Returns a tidy DataFrame with Time_Second injected.
    """
    np.random.seed(seed)
    
    healthy = df[(df['Machine failure'] == 0)].copy()
    fail    = df[(df[failure_col] == 1)].copy()

    # Pick healthy baseline — low stress rows
    baseline = healthy.nsmallest(n_healthy * 3, 'Tool_wear').sample(n_healthy, random_state=seed)

    # Pick approaching rows — healthy rows that are closest to the failure boundary
    if failure_col == 'HDF':
        # Low temp diff AND low RPM = approaching HDF
        healthy['hdf_score'] = (
            (10.0 - (healthy['Process_temperature'] - healthy['Air_temperature'])).clip(lower=0) +
            (1500 - healthy['Rotational_speed']).clip(lower=0) / 100
        )
        approach = healthy.nlargest(n_approach * 2, 'hdf_score').sample(n_approach, random_state=seed)

    elif failure_col == 'OSF':
        # High wear * torque = approaching OSF
        approach = healthy.nlargest(n_approach * 2, 'OSF_proximity').sample(n_approach, random_state=seed)

    elif failure_col == 'PWF':
        # Close to power boundaries (< 4000W or > 8000W)
        healthy['power'] = healthy['Torque'] * (healthy['Rotational_speed'] * 2 * np.pi / 60)
        healthy['pwf_score'] = healthy['power'].apply(
            lambda p: max(0, 4000 - p) + max(0, p - 8000)
        )
        approach = healthy.nlargest(n_approach * 2, 'pwf_score').sample(n_approach, random_state=seed)

    elif failure_col == 'TWF':
        # High tool wear = approaching TWF
        approach = healthy.nlargest(n_approach * 2, 'Tool_wear').sample(n_approach, random_state=seed)

    else:
        approach = healthy.sample(n_approach, random_state=seed)

    # Actual failure rows (capped at available)
    actual_fail = fail.sample(min(n_fail, len(fail)), random_state=seed)

    # Combine: baseline → approach → failure
    scenario = pd.concat([baseline, approach, actual_fail], ignore_index=True)

    # Inject clean time axis and ground truth
    scenario['Time_Second'] = range(len(scenario))
    scenario['Machine_failure'] = scenario['Machine failure'].fillna(0).astype(int)

    keep = ['Time_Second', 'Machine_Type', 'Air_temperature', 'Process_temperature',
            'Rotational_speed', 'Torque', 'Tool_wear', 'Machine_failure']
    return scenario[keep]


def build_synthetic_linear_hdf(n_steps=200):
    """
    Creates a realistic, slow linear degradation that eventually triggers an HDF failure.
    The model has never seen this specific continuous sequence.
    """
    np.random.seed(42)
    data = []
    air_temp = 298.0
    proc_temp = 309.5
    rpm = 1500.0
    torque = 40.0
    wear = 10.0
    
    for t in range(n_steps):
        temp_diff = proc_temp - air_temp
        is_hdf = 1 if (temp_diff < 8.6 and rpm < 1380) else 0
        
        data.append({
            'Time_Second': t,
            'Machine_Type': 'H (High)',
            'Air_temperature': air_temp + np.random.normal(0, 0.05),
            'Process_temperature': proc_temp + np.random.normal(0, 0.05),
            'Rotational_speed': rpm + np.random.normal(0, 2.0),
            'Torque': torque + np.random.normal(0, 0.5),
            'Tool_wear': wear + np.random.normal(0, 0.2),
            'Machine_failure': is_hdf
        })
        
        # Linearly degrade over time towards HDF
        air_temp += 0.015
        proc_temp -= 0.01
        rpm -= 0.8
        wear += 0.2
        torque += 0.05
        
    return pd.DataFrame(data)

def main():
    df = load_and_prep()
    scenarios = {
        'scenario_hdf': ('HDF', 'Heat Dissipation Failure'),
        'scenario_osf': ('OSF', 'Overstrain Failure'),
        'scenario_pwf': ('PWF', 'Power Failure'),
        'scenario_twf': ('TWF', 'Tool Wear Failure'),
    }

    for fname, (col, label) in scenarios.items():
        sc = build_scenario(df, col)
        sc.to_csv(f'{fname}.csv', index=False)
        fails = sc['Machine_failure'].sum()
        print(f"✅ {fname}.csv — {len(sc)} rows | {fails} ground-truth failures | {label}")
        
    # Generate the linear mixed scenario
    mixed_sc = build_synthetic_linear_hdf()
    mixed_sc.to_csv('scenario_mixed.csv', index=False)
    fails_mixed = mixed_sc['Machine_failure'].sum()
    print(f"✅ scenario_mixed.csv — {len(mixed_sc)} rows | {fails_mixed} ground-truth failures | Linear Degradation (HDF) Demo")

if __name__ == '__main__':
    main()
