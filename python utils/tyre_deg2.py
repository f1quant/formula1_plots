import pandas as pd, my_f1_utils, scipy.optimize, itertools, numpy as np, math
import statsmodels.formula.api as smf

def main():
    df = my_f1_utils.load_df()
    df = my_f1_utils.remove_nas(df)
    df = df[df["session_type"]=="R"]
    df = df[(df["year"] == 2024) & (df["round_no"] == 22)]
    fuel_effect = 0.07
    deg_by_compound = {
        "HARD": 0.12,
        "MEDIUM": 0.22,
        "SOFT": 0.40,
        "INTERMEDIATE": 0.10,
        "WET": 0.08
    }

    for (year,round_no), session_df in df.groupby(["year","round_no"]):
        circuit = my_f1_utils.get_uniq(session_df, "circuit_name")
        sched_laps = my_f1_utils.get_uniq(session_df, "sched_laps")
        session_df = my_f1_utils.quick_laps(session_df, sched_laps=sched_laps+2, dist_thresh=3, start_lap_num=2)
        session_df["lap_time1"] = session_df["lap_LapTime"] - fuel_effect * (sched_laps - session_df["lap_LapNumber"])
        session_df["lap_time2"] = session_df["lap_time1"] - session_df["lap_TyreLife"] * session_df["lap_Compound"].map(deg_by_compound)
        for (driver,), driver_df in session_df.groupby(["lap_Driver"]):
            driver_avgs = {x:None for x in deg_by_compound.keys()}
            for (compound,), comp_df in driver_df.groupby(["lap_Compound"]):
                driver_avgs[compound] = comp_df["lap_time2"].mean()
            print(f"{driver}|{'|'.join([str(driver_avgs[c]) for c in ['HARD','MEDIUM','SOFT','INTERMEDIATE','WET']])}")


main()