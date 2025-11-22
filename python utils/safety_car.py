import my_f1_utils, numpy as np
df = my_f1_utils.load_df()

# loop through each race

race_count_by_lap_num = {}
sc_start_count_by_lap_num = {}
red_flag_start_count_by_lap_num = {}

for (year, round_no), session_df in df.groupby(["year", "round_no"]):
    # if year != 2024 or round_no != 21: continue

    race_df = session_df[session_df["session_type"] == "R"]
    circuit_name = my_f1_utils.get_uniq(race_df, "circuit_name")
    race_df = race_df[race_df["lap_TrackStatus"].notna()]
    status_by_lap = {}
    for ri, row in race_df.iterrows():
        lap_num = row["lap_LapNumber"]
        status = row["lap_TrackStatus"]
        if lap_num not in status_by_lap: status_by_lap[lap_num] = {} 
        for s in list(status): status_by_lap[lap_num][s] = 0

    laps_run = int(max(status_by_lap.keys()))
    status_str = ["".join(status_by_lap.get(lap, {}).keys()) for lap in range(1, laps_run + 1)]

    # find contiguous blocks of safety car / vsc, as represented by "4" or "6" or "7"
    sc_periods = []
    in_sc = False
    reg_flag_periods = []
    in_red_flag = False
    for lap in range(1, laps_run + 1):
        status = status_by_lap.get(lap, {})
        has_sc = any(s in status for s in ['4','6','7'])
        if has_sc and not in_sc:
            sc_start = lap
            in_sc = True
        elif not has_sc and in_sc:
            sc_end = lap - 1
            sc_periods.append((sc_start, sc_end))
            in_sc = False

        has_reg_flag = '5' in status
        if has_reg_flag and not in_red_flag:
            rf_start = lap
            in_red_flag = True
        elif not has_reg_flag and in_red_flag:
            rf_end = lap - 1
            reg_flag_periods.append((rf_start, rf_end))
            in_red_flag = False
        
    if in_sc:
        sc_end = laps_run
        sc_periods.append((sc_start, sc_end))
    if in_red_flag:
        rf_end = laps_run
        reg_flag_periods.append((rf_start, rf_end))

    for lap in range(1, laps_run + 1):
        in_sc = any(start < lap <= end for (start, end) in sc_periods)
        in_rf = any(start < lap <= end for (start, end) in reg_flag_periods)
        if not in_sc and not in_rf:
            race_count_by_lap_num[lap] = race_count_by_lap_num.get(lap, 0) + 1
    
    for (start, end) in sc_periods:
        sc_start_count_by_lap_num[start] = sc_start_count_by_lap_num.get(start, 0) + 1
    for (start, end) in reg_flag_periods:
        red_flag_start_count_by_lap_num[start] = red_flag_start_count_by_lap_num.get(start, 0) + 1

print("Lap|Races|SC Starts|Red Flag Starts")
for lap in range(1, max(race_count_by_lap_num.keys()) + 1):
    print(f"{lap}|{race_count_by_lap_num.get(lap, 0)}|{sc_start_count_by_lap_num.get(lap, 0)}|{red_flag_start_count_by_lap_num.get(lap, 0)}")
