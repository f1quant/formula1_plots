import my_f1_utils, numpy as np

def main():
    df = my_f1_utils.remove_nas(my_f1_utils.load_df())
    df = df[(df["year"] == 2024) & (df["round_no"] == 22) & (df["session_type"] == "R")] # vegas
    sched_laps = my_f1_utils.get_uniq(df, "sched_laps")
    df = df[df["finish_status"].isin(["Finished","Lapped","Retired"])]
    df = my_f1_utils.filter_safety_car(df)
    df = my_f1_utils.filter_rainfall(df)
    df = df[df["lap_LapNumber"] <= sched_laps - 2]
    df = df[df["lap_LapNumber"] >= 3]
    df = df[df["lap_PitInTime"].isna()]
    df = df[df["lap_PitOutTime"].isna()]
    
    fuel_effect = -0.08
    df["lap_time_adj"] = df["lap_LapTime"] + fuel_effect * (sched_laps - df["lap_LapNumber"])
    df["speed"] = 1 / df["lap_time_adj"]
    
    for (driver,stint), stint_df in df.groupby(["lap_Driver","lap_Stint"]):
        stint = int(stint)
        if driver != "RUS" or stint != 3: continue

        stint_df = stint_df.sort_values(by="lap_LapNumber")
        stint_df['cum_speed'] = stint_df['speed'].cumsum().shift(fill_value=0)
        stint_df['cum_speed_squared'] = (stint_df['cum_speed'] ** 2)
        stint_df = stint_df[stint_df["min_dist"] > 5]
        if len(stint_df) < 6: continue # too short to analyze
        compound = my_f1_utils.get_uniq(stint_df, "lap_Compound")
        
        xs = stint_df['cum_speed_squared']
        ys = stint_df["lap_time_adj"]
        corrx = xs.corr(stint_df["lap_TyreLife"])
        for row in stint_df.itertuples():
            print(f"{row.lap_LapNumber}|{row.lap_TyreLife}|{row.lap_LapTime}|{row.lap_time_adj}|{row.speed}|{row.cum_speed}")

        corr = xs.corr(ys)
        beta = float(np.std(ys)) / float(np.std(xs)) * corr
        alpha = ys.mean() - beta * xs.mean()
        rmse = float(np.sqrt(np.mean((ys - (alpha + beta * xs)) ** 2)))
        # print(f"{driver}|{stint}|{compound}|{len(stint_df)}|{alpha}|{beta}|{corr}|{rmse}|{corrx}")

main()