import pandas as pd, my_f1_utils, scipy.optimize, itertools, numpy as np, math
import statsmodels.formula.api as smf

def group_stints(stints):
    stints_by_driver_compound = {}
    for stint in stints:
        driver = stint["driver"]
        compound = stint["compound"]
        if driver not in stints_by_driver_compound:
            stints_by_driver_compound[driver] = {}
        if compound not in stints_by_driver_compound[driver]:
            stints_by_driver_compound[driver][compound] = []
        stints_by_driver_compound[driver][compound].append(stint)
    return stints_by_driver_compound

def stint_variation(fuel_effect,grouped_stints):
    stint_diffs = []
    for driver, driver_stints in grouped_stints.items():
        for compound, stints in driver_stints.items():
            if len(stints) < 2: continue
            alphas = [my_f1_utils.adjust_stint_regression(x, fuel_effect)[0] for x in stints]
            stint_diffs += [x - alphas[0] for x in alphas[1:]]
            # stint_diffs += list(np.diff(alphas))
                
    if len(stint_diffs) < 5:
        raise Exception("Not enough stints to analyze")
    var = math.sqrt(sum([x**2 for x in stint_diffs]) / len(stint_diffs))
    return var

def main():
    df = my_f1_utils.load_df()
    df = my_f1_utils.remove_nas(df)
    # brazil_races = [(2024,21), (2023,20), (2022, 21), (2021, 19)]
    df = df[(df["year"] == 2024) & (df["round_no"] == 22)]
    
    track_stats = []
    for (year,round_no), meeting_df in df.groupby(["year","round_no"]):
        circuit_name = my_f1_utils.get_uniq(meeting_df, "circuit_name")
        lap_length = meeting_df["lap_length"].mean()
        lap_time = my_f1_utils.quick_laps(meeting_df,1000,["Finished"])["lap_LapTime"].mean()
        avg_speed = lap_length / lap_time * 3.6 # km/h

        # track temp in the first 10 laps
        race_df = meeting_df[meeting_df["session_type"] == "R"]
        sched_laps = my_f1_utils.get_uniq(race_df, "sched_laps")
        temp_start = race_df[race_df["lap_LapNumber"] <= 10]["weather_TrackTemp"].mean()
        temp_end = race_df[race_df["lap_LapNumber"] >= sched_laps - 10]["weather_TrackTemp"].mean()

        try:
            stints = my_f1_utils.stint_regressions(meeting_df)
            grouped_stints = group_stints(stints)
            test = stint_variation(-0.02,grouped_stints)
        except Exception as e:
            # print(f"Error processing {year}|{round_no}: {e}")
            continue
        result = scipy.optimize.minimize(lambda x: stint_variation(x[0],grouped_stints), 0.0, bounds=[(-5.0, 5.0)])
        fuel_effect = result.x[0]
        var = stint_variation(fuel_effect,grouped_stints)
        print(f"{year}|{round_no}|{circuit_name}|{lap_length}|{avg_speed}|{temp_start}|{temp_end}|{temp_end-temp_start}|{fuel_effect}|{var}")
        
        # track_stats.append({
        #     "year": year,
        #     "round_no": round_no,
        #     "circuit_name": circuit_name,
        #     "lap_length": lap_length/1e3,
        #     "avg_speed": avg_speed/1e3,
        #     "temp_start": temp_start,
        #     "temp_end": temp_end,
        #     "temp_delta": (temp_end - temp_start)/1e3,
        #     "fuel_effect": fuel_effect
        # })

    # track_stats_df = pd.DataFrame(track_stats)

    # ols_formula = "fuel_effect ~ lap_length + avg_speed + temp_delta"
    # res_ols = smf.ols(ols_formula, data=track_stats_df).fit()    
    # print(res_ols.summary())
    # rmse = float(np.sqrt(np.mean((track_stats_df["fuel_effect"] - res_ols.fittedvalues) ** 2)))
    # print(f"RMSE: {rmse}")

main()