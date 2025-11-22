import my_f1_utils, numpy as np

def calc_gaps(df):
    sched_laps = my_f1_utils.get_uniq(df, "sched_laps")
    df = df[~df["lap_Position"].isna()]     # filter our laps with position na, those are dnfs
    gaps_by_lap = []
    for lap in range(1, sched_laps + 1):
        lap_df = df[df["lap_LapNumber"] == lap].sort_values(by=["lap_Position"])
        drivers = list(lap_df["lap_Driver"].values)
        lap_df = lap_df.set_index("lap_Driver")
        first_driver_time = lap_df.at[drivers[0], "lap_Time"]
        finish_times = [[drv, lap_df.at[drv, "lap_Time"] - first_driver_time] for drv in drivers]
        gaps_by_lap.append(finish_times)
    return gaps_by_lap

def main():
    all_df = my_f1_utils.load_df()
    year, round_no, session_type = 2024, 22, "R"
    df = all_df[(all_df["year"] == year) & (all_df["round_no"] == round_no) & (all_df["session_type"] == session_type)]
    gaps_by_lap = calc_gaps(df) # for each lap, list of [driver, gap to leader]
    
    

main()