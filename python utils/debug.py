import my_f1_utils, numpy as np
df = my_f1_utils.load_df()
df = df[(df["year"]==2025) & (df["round_no"]==1) & (df["session_type"]=="R")]

# group by lap number
for lap_num, lap_df in df.groupby("lap_LapNumber"):
    # how many rows in lap_df have weather_Rainfall set to true?
    rainfall_df = lap_df[lap_df["weather_Rainfall"]==True]
    print(f"Lap {lap_num}: drivers: {len(lap_df)}, Rainfall: {len(rainfall_df)}")