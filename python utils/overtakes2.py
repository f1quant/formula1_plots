import my_f1_utils, matplotlib.pyplot as plt, numpy as np, math
import matplotlib.ticker as mtick

def calc_gaps(df):
    sched_laps = my_f1_utils.get_uniq(df, "sched_laps")
    df = df[~df["lap_Position"].isna()]     # filter our laps with position na, those are dnfs
    gaps_by_lap = []
    for lap in range(1, sched_laps + 1):
        lap_df = df[df["lap_LapNumber"] == lap].sort_values(by=["lap_Position"])
        drivers = list(lap_df["lap_Driver"].values)
        lap_df = lap_df.set_index("lap_Driver")
        finish_times = [lap_df.at[drv, "lap_Time"] for drv in drivers]
        gaps_by_lap.append(np.diff(finish_times))
    return gaps_by_lap

def main():
    races = [(2025,x) for x in range(1, 21)]
    races = [(2024,x) for x in range(1, 25)]
    all_df = my_f1_utils.load_df()

    # make a grid of all the races. Make them share y axis
    ncols = min(5, len(races))
    nrows = math.ceil(len(races)/ncols)
    plt.rcParams.update({'font.size': 5})
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 6), sharey=True, constrained_layout=False)
    # fig.set_constrained_layout_pads(w_pad=0, h_pad=0, wspace=0, hspace=0)

    session_type = "R"
    for ri, (year, round_no) in enumerate(races):
        df = all_df[(all_df["year"] == year) & (all_df["round_no"] == round_no) & (all_df["session_type"] == session_type)]
        circuit_name = my_f1_utils.get_uniq(df, "circuit_name")
        gaps = calc_gaps(df)
        
        # flattan gaps using numpy and make a histogram in perceptions of all gaps
        all_gaps = np.concatenate(gaps)
        hist, bin_edges = np.histogram(all_gaps, bins=list(np.arange(0,5)))
        hist = np.cumsum([h/len(all_gaps) for h in hist])

        ax = axs[ri // ncols, ri % ncols] if nrows > 1 else axs[ri % ncols]
        ax.bar(bin_edges[:-1], hist, width=1, align='edge', edgecolor='black')
        ax.set_title(f"{year}.{round_no} {circuit_name}")
        ax.set_xticks(bin_edges)

    for ax in axs.flat:
        ax.tick_params(axis="y", which="both", left=True, right=True, labelleft=True)
        ax.yaxis.set_major_locator(mtick.MultipleLocator(0.1))                 # 0.0, 0.1, …, 1.0
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))

    plt.tight_layout()
    plt.show()
    
main()
