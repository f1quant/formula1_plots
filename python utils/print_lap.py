import fastf1
import my_f1_utils # cache

# session = fastf1.get_session(2024, 22, "R")
# session.load()
# laps = session.laps.pick_drivers("RUS")
# lap = laps[laps["LapNumber"] == 5]
# # lap = laps[laps["LapNumber"] == 40]

session = fastf1.get_session(2024, 22, "Q")
session.load()
laps = session.laps.split_qualifying_sessions()[2]
driver_laps = laps.pick_drivers("RUS")
lap = driver_laps.pick_fastest()

car_data = lap.get_car_data(pad=1, pad_side='both')
car_t = car_data["Time"].dt.total_seconds().to_numpy()
car_speed = car_data["Speed"].to_numpy()

# for row in car_data.itertuples():
    # print(f"{row.Time.total_seconds()}|{row.Speed}|{row.Throttle}|{row.Brake}")

pos_data = lap.get_pos_data(pad=1, pad_side='both')
pos_coords = [pos_data[c].to_numpy()/10 for c in ["X", "Y", "Z"]]
pos_t = pos_data["Time"].dt.total_seconds().to_numpy()
for idx in range(len(pos_t)):
    print(f"{pos_t[idx]}|{pos_coords[0][idx]}|{pos_coords[1][idx]}|{pos_coords[2][idx]}")