# trying to use the break point
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import fastf1
import my_f1_utils # cache

def negative_log_likelihood(params, X, y):
    beta0, beta1 = params
    sigmoid = lambda t: 1.0 / (1.0 + np.exp(-np.clip((t-beta0)*beta1, -500, 500)))
    p = sigmoid(X)
    epsilon = 1e-12 
    p = np.clip(p, epsilon, 1 - epsilon)
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

def main():
    events = [
        {"Type": "Break", "minT": 1.00, "maxT": 3.00},
        {"Type": "Throttle", "minT": 3.50, "maxT": 5.00},
        {"Type": "Break", "minT": 19.50, "maxT": 21.00},
        {"Type": "Throttle", "minT": 21.50, "maxT": 24.00},
        {"Type": "Break", "minT": 28.00, "maxT": 30.70},
        {"Type": "Throttle", "minT": 30.70, "maxT": 33.00},
        {"Type": "Break", "minT": 34.00, "maxT": 36.40},
        {"Type": "Throttle", "minT": 36.40, "maxT": 39.00},
        {"Type": "Break", "minT": 49.00, "maxT": 51.80},
        {"Type": "Throttle", "minT": 51.80, "maxT": 55.00},
        {"Type": "Break", "minT": 74.00, "maxT": 77.00},
        {"Type": "Throttle", "minT": 77.00, "maxT": 80.00},
    ]

    session = fastf1.get_session(2024, 22, "Q")
    session.load()
    laps = session.laps.split_qualifying_sessions()[2]
    for driver in ["RUS","SAI"]:
        driver_laps = laps.pick_drivers(driver)    
        lap = driver_laps.pick_fastest()
        car_data = lap.get_car_data(pad=1, pad_side='both')
        data = np.stack((car_data["Time"].dt.total_seconds().to_numpy(), car_data["Brake"].to_numpy().astype(int)), axis=1)
        for event in events:
            event_data = data[(data[:,0] >= event["minT"]) & (data[:,0] <= event["maxT"])]
            if event["Type"] == "Throttle":
                event_data[:,1] = 1 - event_data[:,1]
            
            # print(event_data)
            result = scipy.optimize.minimize(
                negative_log_likelihood,
                [(event["minT"] + event["maxT"])/2 , 1.0],
                args=(event_data[:, 0], event_data[:, 1]),
                method='L-BFGS-B'
            )

            assert result.success
            est_t, beta = result.x
            assert event["minT"] <= est_t <= event["maxT"]
            print(f"{driver}|{event['Type']}|{event['minT']}|{event['maxT']}|{est_t}|{beta}")

def err_func(t_delta, d1_data, d2_data):
    t_delta = t_delta[0]
    d1_interp_speeds = np.interp(d2_data[:,0] + t_delta, d1_data[:,0], d1_data[:,1])
    speed_diffs = d2_data[:,1] - d1_interp_speeds
    ans = np.sum(speed_diffs ** 2)
    return ans

# doesn't work on straights
# def err_func(args, d1_data, d2_data):
#     t_delta, speed_shift, speed_scale = args
#     d1_interp_speeds = np.interp(d2_data[:,0] + t_delta, d1_data[:,0], d1_data[:,1])
#     speed_diffs = (d2_data[:,1]+speed_shift)*speed_scale - d1_interp_speeds
#     ans = np.sum(speed_diffs ** 2)
#     return ans

def data_from_lap(lap):
    car_data = lap.get_car_data(pad=1, pad_side='both')
    return np.stack((car_data["Time"].dt.total_seconds().to_numpy(), car_data["Speed"].to_numpy()), axis=1)

def print_sector_times(lap):
    sector_times = [
        lap["Sector1Time"].total_seconds(),
        lap["Sector2Time"].total_seconds(),
        lap["Sector3Time"].total_seconds(),
    ]
    print(f"{'|'.join(map(str, sector_times))}")

def main2():

    events = [
        [0.00, 6.00], # breaking turn 1
        [19.0, 22.00], # breaking turn 4
        [25.0, 27.0], # mini straight end of sector 1
        [27.00, 31.00], # breaking turn 7
        [31.00, 40.00], # sphere
        [48.00, 52.00], # turn into the strip
        [51.00, 57.00], # strip traction sector 2->3
        [72.00, 83.00], # turn into final straight
        [85.00, 90.00], # final wiggle
    ]

    session = fastf1.get_session(2024, 22, "Q")
    session.load()
    laps = session.laps.split_qualifying_sessions()[2]
    driver1 = "RUS"
    driver2s = ["SAI", "GAS", "LEC", "VER", "NOR", "TSU", "PIA", "HUL"]

    my_laps = {}
    for driver2 in driver2s:
        for driver in [driver1, driver2]:
            my_laps[driver] = laps.pick_drivers(driver).pick_fastest()

    session = fastf1.get_session(2024, 22, "R")
    session.load()
    laps = session.laps.pick_drivers("RUS")
    my_laps["RUS-5"] = laps[laps["LapNumber"] == 5].iloc[0]
    my_laps["RUS-40"] = laps[laps["LapNumber"] == 40].iloc[0]
    driver2s.append("RUS-5")
    driver2s.append("RUS-40")

    driver1_data = data_from_lap(my_laps[driver1])
    print_sector_times(my_laps[driver1])
    results = []
    for driver2 in driver2s:
        driver2_data = data_from_lap(my_laps[driver2])
        print_sector_times(my_laps[driver2])
        row = []
        for [minT,maxT] in events:
            d1_data = driver1_data[(driver1_data[:,0] >= minT) & (driver1_data[:,0] <= maxT)]
            d2_data = driver2_data[(driver2_data[:,0] >= minT) & (driver2_data[:,0] <= maxT)]

            result = scipy.optimize.minimize(
                err_func,
                [0.0],
                args=(d2_data, d1_data),
                method='L-BFGS-B',
                options = {"ftol": 1e-6, "gtol": 1e-6, "maxls": 50},
            )

            if not result.success:
                row.append(result.message)
            else:
                row.append(result.x[0])
        results.append(row)

    results = np.array(results).T
    # for ei,[minT, maxT] in enumerate(events):
    #     print(f"{minT}|{maxT}", end="|")
    #     print("|".join(map(str, results[ei])))
        

main2()