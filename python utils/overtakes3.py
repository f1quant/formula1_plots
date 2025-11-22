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

def find_consecutive_pairs(gaps_by_lap):
    ans = []
    active_pairs = {}  # key: (driver_front, driver_behind), value: {from_lap, gaps}

    for lap_num, lap_data in enumerate(gaps_by_lap, start=1):
        current_pairs = set() # lap_data is list of [driver, gap_to_leader]
        for i in range(len(lap_data) - 1):
            gap_between = lap_data[i + 1][1] - lap_data[i][1]

            pair_key = (lap_data[i][0], lap_data[i + 1][0])
            current_pairs.add(pair_key)

            if pair_key in active_pairs:
                # Continue existing sequence
                active_pairs[pair_key]["gaps"].append(gap_between)
            else:
                # Start new sequence
                active_pairs[pair_key] = {
                    "driver_front": pair_key[0],
                    "driver_behind": pair_key[1],
                    "from_lap": lap_num,
                    "gaps": [gap_between]
                }

        # Check which pairs are no longer consecutive and finalize them
        pairs_to_remove = []
        for pair_key in active_pairs:
            if pair_key not in current_pairs:
                ans.append(active_pairs[pair_key]) # This pair is no longer consecutive, save it
                pairs_to_remove.append(pair_key)

        for pair_key in pairs_to_remove:
            del active_pairs[pair_key]

    for pair_data in active_pairs.values():
        ans.append(pair_data)

    return ans

def main():
    all_df = my_f1_utils.load_df()
    year, round_no, session_type = 2024, 22, "R"
    df = all_df[(all_df["year"] == year) & (all_df["round_no"] == round_no) & (all_df["session_type"] == session_type)]
    gaps_by_lap = calc_gaps(df) # for each lap, list of [driver, gap to leader]

    consecutive_pairs = find_consecutive_pairs(gaps_by_lap)

    for pair in consecutive_pairs:
        if len(pair["gaps"]) < 3: continue
        print(f"{pair['driver_front']} ahead of {pair['driver_behind']} for {len(pair['gaps'])} laps starting from lap {pair['from_lap']}")
        print(f"  Gaps: {[f'{g:.3f}' for g in pair['gaps']]}")

main()