import pandas as pd, my_f1_utils, scipy.optimize, itertools, numpy as np, math
import statsmodels.formula.api as smf

def main():
    df = my_f1_utils.load_df()
    df = my_f1_utils.remove_nas(df)
    df = df[df["session_type"]=="R"]
    for (year,round_no), session_df in df.groupby(["year","round_no"]):
        # print(len(session_df[(session_df["lap_TyreLife"]==7) & (session_df["lap_Compound"]=="MEDIUM")]))
        circuit = my_f1_utils.get_uniq(session_df, "circuit_name")
        sched_laps = my_f1_utils.get_uniq(session_df, "sched_laps")
        session_df = my_f1_utils.quick_laps(session_df, sched_laps=sched_laps+2, dist_thresh=3, start_lap_num=2)
        all_time_diffs = []
        all_lap_diffs = []
        for (driver,compound,life), group_df in session_df.groupby(["lap_Driver","lap_Compound","lap_TyreLife"]):
            if len(group_df) < 2: continue
            lap_times = group_df["lap_LapTime"].values
            lap_numbers = group_df["lap_LapNumber"].values
            time_diffs = np.diff(lap_times)
            all_time_diffs.extend(time_diffs)
            lap_diffs = np.diff(lap_numbers)
            all_lap_diffs.extend(lap_diffs)
            print(f"{driver}|{compound}|{life}|{"|".join(map(str,lap_times))}|{"|".join(map(str,lap_numbers))}")

        if len(all_time_diffs) < 2: 
            print(f"{year}|{round_no}")
            continue

        avg_x, avg_y = np.mean(all_lap_diffs), np.mean(all_time_diffs)
        std_x, std_y = np.std(all_lap_diffs), np.std(all_time_diffs)
        
        if std_x == 0 or std_y == 0: 
            print(f"{year}|{round_no}")
            continue

        corr = np.corrcoef(all_lap_diffs, all_time_diffs)[0,1]
        beta = std_y / std_x * corr
        alpha = avg_y - beta * avg_x
        beta2 = (corr*std_y*std_x + avg_x*avg_y) / (std_x**2 + avg_x**2)
        # print(f"{year}|{round_no}|{circuit}|{alpha}|{beta}|{beta2}|{corr}|{len(all_time_diffs)}")

main()