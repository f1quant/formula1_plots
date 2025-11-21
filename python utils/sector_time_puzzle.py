import fastf1, logging, numpy as np, matplotlib.pyplot as plt
fastf1.Cache.enable_cache("fastf1_cache")
fastf1.logger.set_log_level(logging.ERROR)
fastf1.Cache.offline_mode(True)

year = 2024
round_no = "Las Vegas"
avg_coords = {}
avg_speeds = {}
for i, sess_type in enumerate(["Q","R"]):
    session = fastf1.get_session(year, round_no, sess_type)
    session.load()
    coords = [[],[],[]]
    speeds = [[],[],[]]
    for _,lap in session.laps.iterrows():
        pos_data = lap.get_pos_data(pad=2, pad_side='both')
        sts = [
            lap["Sector1Time"].total_seconds(),
            lap["Sector1Time"].total_seconds()+lap["Sector2Time"].total_seconds(),
            lap["Sector1Time"].total_seconds()+lap["Sector2Time"].total_seconds()+lap["Sector3Time"].total_seconds(),
        ]
        if any(np.isnan(st) for st in sts): continue
        
        # interp_t = lambda t: [np.interp(t, pos_data["Time"].dt.total_seconds(), pos_data[c]/10) for c in ["X","Y","Z"]]
        
        car_data = lap.get_car_data(pad=2, pad_side='both')
        for i in range(3):
            coords[i].append([np.interp(sts[i], pos_data["Time"].dt.total_seconds(), pos_data[c]/10) for c in ["X","Y","Z"]])
            speeds[i].append(np.interp(sts[i], car_data["Time"].dt.total_seconds(), car_data["Speed"]))

    avg_coords[sess_type] = np.mean(np.array(coords), axis=1)
    avg_speeds[sess_type] = np.mean(np.array(speeds), axis=1)

avg_dist = np.sqrt(( avg_coords['R'] - avg_coords['Q'] )[:,0]**2 + ( avg_coords['R'] - avg_coords['Q'] )[:,1]**2 + ( avg_coords['R'] - avg_coords['Q'] )[:,2]**2 )
print(f"{year}|{round_no}|{'|'.join(map(str, avg_dist))}")

print(avg_speeds)