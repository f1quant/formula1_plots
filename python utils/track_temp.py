
import my_f1_utils

df = my_f1_utils.load_df()
df = df[(df["year"] == 2023) & (df["round_no"] == 20)]

for lap in sorted(df["lap_LapNumber"].unique()):
    lap_df = df[df["lap_LapNumber"] == lap]
    avg_temp = lap_df["weather_TrackTemp"].mean()
    perc_rain = (lap_df["weather_Rainfall"] == True).mean()
    perc_safety_car = (lap_df["lap_TrackStatus"].str.contains('[467]')).mean()
    print(f"{lap}|{avg_temp:.2f}|{perc_rain:.4f}|{perc_safety_car:.4f}")