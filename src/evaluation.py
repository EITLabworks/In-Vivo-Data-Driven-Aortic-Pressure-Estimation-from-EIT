import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import shapely.geometry as shape_geom
import pandas as pd
import seaborn as sns


# color maps
C_MAP = {
    "python": {
        "E": "C3",
        "D": "C1",
        "C": "C4",
        "B": "C0",
        "A": "C2",
    },
    "saugel": {
        "E": "C1",
        "D": "yellow",
        "C": "C8",
        "B": "C9",
        "A": "C0",
    },
    "risklvl": {
        "E": "#811B15",
        "D": "#FF3619",
        "C": "#FF8E2A",
        "B": "#FEC433",
        "A": "#34AF4A",
    },
}


MAP = {
    "E1": [[0, 10], [20, 30], [20, 70], [35, 70], [60, 130], [60, 300]],
    "D1": [[20, 60], [40, 60], [75, 150], [75, 300]],
    "C1": [[20, 55], [45, 55], [100, 150], [100, 300]],
    "B1": [[20, 50], [48, 50], [120, 140], [120, 300]],
    "B2": [[50, 0], [50, 35], [140, 120], [300, 120]],
    "C2": [[65, 0], [65, 35], [145, 100], [300, 100]],
    "D2": [[90, 0], [90, 45], [145, 85], [300, 85]],
    "E2": [[120, 0], [120, 50], [140, 60], [300, 60]],
}

# SAP line coordinates
SAP = {
    "E1": [[0, 20], [35, 70], [35, 110], [60, 110], [80, 180], [80, 300]],
    "D1": [[35, 85], [60, 85], [110, 210], [110, 300]],
    "C1": [[35, 78], [65, 78], [160, 215], [160, 300]],
    "B1": [[35, 70], [65, 70], [190, 220], [190, 300]],
    "B2": [[80, 0], [80, 65], [115, 90], [140, 90], [220, 190], [300, 190]],
    "C2": [[100, 0], [100, 70], [145, 70], [210, 160], [300, 160]],
    "D2": [[150, 0], [150, 60], [210, 140], [300, 140]],
    "E2": [[180, 0], [180, 65], [210, 85], [300, 85]],
}

MAP_V = {
    "E1": [[0, 10], [20, 30], [20, 70], [35, 70], [60, 130], [60, 300], [0, 300]],
    "D1": [
        [20, 60],
        [40, 60],
        [75, 150],
        [75, 300],
        [60, 300],
        [60, 130],
        [35, 70],
        [20, 70],
    ],
    "C1": [
        [20, 55],
        [45, 55],
        [100, 150],
        [100, 300],
        [75, 300],
        [75, 150],
        [40, 60],
        [20, 60],
    ],
    "B1": [
        [20, 50],
        [48, 50],
        [120, 140],
        [120, 300],
        [120, 300],
        [100, 300],
        [100, 150],
        [45, 55],
        [20, 55],
    ],
    "A1": [
        [0, 0],
        [0, 10],
        [20, 30],
        [20, 50],
        [48, 50],
        [120, 140],
        [120, 300],
        [300, 300],
        [300, 120],
        [140, 120],
        [50, 35],
        [50, 0],
    ],
    "B2": [
        [50, 0],
        [50, 35],
        [140, 120],
        [300, 120],
        [300, 100],
        [145, 100],
        [65, 35],
        [65, 0],
    ],
    "C2": [
        [65, 0],
        [65, 35],
        [145, 100],
        [300, 100],
        [300, 85],
        [145, 85],
        [90, 45],
        [90, 0],
    ],
    "D2": [
        [90, 0],
        [90, 45],
        [145, 85],
        [300, 85],
        [300, 60],
        [140, 60],
        [120, 50],
        [120, 0],
    ],
    "E2": [[120, 0], [120, 50], [140, 60], [300, 60], [300, 0]],
}

# SAP vertices
SAP_V = {
    "E1": [[0, 20], [35, 70], [35, 110], [60, 110], [80, 180], [80, 300], [0, 300]],
    "D1": [
        [35, 85],
        [60, 85],
        [110, 210],
        [110, 300],
        [80, 300],
        [80, 180],
        [60, 110],
        [35, 110],
    ],
    "C1": [
        [35, 78],
        [65, 78],
        [160, 215],
        [160, 300],
        [110, 300],
        [110, 210],
        [60, 85],
        [35, 85],
    ],
    "B1": [
        [35, 70],
        [65, 70],
        [190, 220],
        [190, 300],
        [160, 300],
        [160, 215],
        [65, 78],
        [35, 78],
    ],
    "A1": [
        [80, 0],
        [80, 65],
        [115, 90],
        [140, 90],
        [220, 190],
        [300, 190],
        [300, 300],
        [190, 300],
        [190, 220],
        [65, 70],
        [35, 70],
        [0, 20],
        [0, 0],
    ],
    "B2": [
        [80, 0],
        [80, 65],
        [115, 90],
        [140, 90],
        [220, 190],
        [300, 190],
        [300, 160],
        [210, 160],
        [145, 70],
        [100, 70],
        [100, 0],
    ],
    "C2": [
        [100, 0],
        [100, 70],
        [145, 70],
        [210, 160],
        [300, 160],
        [300, 140],
        [210, 140],
        [150, 60],
        [150, 0],
    ],
    "D2": [
        [150, 0],
        [150, 60],
        [210, 140],
        [300, 140],
        [300, 85],
        [210, 85],
        [180, 65],
        [180, 0],
    ],
    "E2": [[180, 0], [180, 65], [210, 85], [300, 85], [300, 0]],
}


def DAP_SAP_MAP_kde(
    y_pred,
    y_true,
    sname="none",
    colormap="python",
    dot_size=4,
):
    X = y_true
    Y = y_pred
    # set layout-theme
    sns.set_theme(style="white", context="talk")
    plt.rcParams.update({"font.size": 12})
    plt.rcParams["axes.grid"] = True
    sns.set(font_scale=1.2)

    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    sns.set_context(context="paper", font_scale=1.4)
    # kde levels:
    levels = 5
    AP_factors = {"MAP": 160, "SAP": 180, "DAP": 180}

    MAP = {
        "E1": [[0, 10], [20, 30], [20, 70], [35, 70], [60, 130], [60, 300]],
        "D1": [[20, 60], [40, 60], [75, 150], [75, 300]],
        "C1": [[20, 55], [45, 55], [100, 150], [100, 300]],
        "B1": [[20, 50], [48, 50], [120, 140], [120, 300]],
        "B2": [[50, 0], [50, 35], [140, 120], [300, 120]],
        "C2": [[65, 0], [65, 35], [145, 100], [300, 100]],
        "D2": [[90, 0], [90, 45], [145, 85], [300, 85]],
        "E2": [[120, 0], [120, 50], [140, 60], [300, 60]],
    }

    # SAP line coordinates
    SAP = {
        "E1": [[0, 20], [35, 70], [35, 110], [60, 110], [80, 180], [80, 300]],
        "D1": [[35, 85], [60, 85], [110, 210], [110, 300]],
        "C1": [[35, 78], [65, 78], [160, 215], [160, 300]],
        "B1": [[35, 70], [65, 70], [190, 220], [190, 300]],
        "B2": [[80, 0], [80, 65], [115, 90], [140, 90], [220, 190], [300, 190]],
        "C2": [[100, 0], [100, 70], [145, 70], [210, 160], [300, 160]],
        "D2": [[150, 0], [150, 60], [210, 140], [300, 140]],
        "E2": [[180, 0], [180, 65], [210, 85], [300, 85]],
    }

    MAP_V = {
        "E1": [[0, 10], [20, 30], [20, 70], [35, 70], [60, 130], [60, 300], [0, 300]],
        "D1": [
            [20, 60],
            [40, 60],
            [75, 150],
            [75, 300],
            [60, 300],
            [60, 130],
            [35, 70],
            [20, 70],
        ],
        "C1": [
            [20, 55],
            [45, 55],
            [100, 150],
            [100, 300],
            [75, 300],
            [75, 150],
            [40, 60],
            [20, 60],
        ],
        "B1": [
            [20, 50],
            [48, 50],
            [120, 140],
            [120, 300],
            [120, 300],
            [100, 300],
            [100, 150],
            [45, 55],
            [20, 55],
        ],
        "A1": [
            [0, 0],
            [0, 10],
            [20, 30],
            [20, 50],
            [48, 50],
            [120, 140],
            [120, 300],
            [300, 300],
            [300, 120],
            [140, 120],
            [50, 35],
            [50, 0],
        ],
        "B2": [
            [50, 0],
            [50, 35],
            [140, 120],
            [300, 120],
            [300, 100],
            [145, 100],
            [65, 35],
            [65, 0],
        ],
        "C2": [
            [65, 0],
            [65, 35],
            [145, 100],
            [300, 100],
            [300, 85],
            [145, 85],
            [90, 45],
            [90, 0],
        ],
        "D2": [
            [90, 0],
            [90, 45],
            [145, 85],
            [300, 85],
            [300, 60],
            [140, 60],
            [120, 50],
            [120, 0],
        ],
        "E2": [[120, 0], [120, 50], [140, 60], [300, 60], [300, 0]],
    }

    # SAP vertices
    SAP_V = {
        "E1": [[0, 20], [35, 70], [35, 110], [60, 110], [80, 180], [80, 300], [0, 300]],
        "D1": [
            [35, 85],
            [60, 85],
            [110, 210],
            [110, 300],
            [80, 300],
            [80, 180],
            [60, 110],
            [35, 110],
        ],
        "C1": [
            [35, 78],
            [65, 78],
            [160, 215],
            [160, 300],
            [110, 300],
            [110, 210],
            [60, 85],
            [35, 85],
        ],
        "B1": [
            [35, 70],
            [65, 70],
            [190, 220],
            [190, 300],
            [160, 300],
            [160, 215],
            [65, 78],
            [35, 78],
        ],
        "A1": [
            [80, 0],
            [80, 65],
            [115, 90],
            [140, 90],
            [220, 190],
            [300, 190],
            [300, 300],
            [190, 300],
            [190, 220],
            [65, 70],
            [35, 70],
            [0, 20],
            [0, 0],
        ],
        "B2": [
            [80, 0],
            [80, 65],
            [115, 90],
            [140, 90],
            [220, 190],
            [300, 190],
            [300, 160],
            [210, 160],
            [145, 70],
            [100, 70],
            [100, 0],
        ],
        "C2": [
            [100, 0],
            [100, 70],
            [145, 70],
            [210, 160],
            [300, 160],
            [300, 140],
            [210, 140],
            [150, 60],
            [150, 0],
        ],
        "D2": [
            [150, 0],
            [150, 60],
            [210, 140],
            [300, 140],
            [300, 85],
            [210, 85],
            [180, 65],
            [180, 0],
        ],
        "E2": [[180, 0], [180, 65], [210, 85], [300, 85], [300, 0]],
    }

    # color maps
    C_MAP = {
        "python": {
            "E": "C3",
            "D": "C1",
            "C": "C4",
            "B": "C0",
            "A": "C2",
        },
        "saugel": {
            "E": "C1",
            "D": "yellow",
            "C": "C8",
            "B": "C9",
            "A": "C0",
        },
        "risklvl": {
            "E": "#811B15",
            "D": "#FF3619",
            "C": "#FF8E2A",
            "B": "#FEC433",
            "A": "#34AF4A",
        },
    }

    fig, ax = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    ax[0].set_ylabel("Model (mmHg)")

    # DAP
    s = 0

    polygon = Polygon(
        [[0, 0], [0, 300], [300, 300], [300, 0]],
        closed=True,
        facecolor=C_MAP[colormap]["A"],
        alpha=0.4,
        zorder=1,
    )
    ax[s].add_patch(polygon)

    ax[s].set_title("DAP")
    ax[s].scatter(
        x=X[:, 0],
        y=Y[:, 0],
        s=dot_size,
        c="k",
    )
    sns.kdeplot(
        x=X[:, 0],
        y=Y[:, 0],
        cmap="rocket",
        fill=True,
        # clip=(-5, 5),
        # cut=10,
        # thresh=0,
        levels=levels,  # levels=[0.2, 0.4, 0.6,0.8,1],
        ax=ax[s],
    )
    ax[s].set_xlim(0, 200)
    ax[s].set_ylim(0, 200)
    ax[s].set_xlabel("Gold Standard (mmHg)")
    ax[s].plot([0, 300], [0, 300], linestyle="--", color="black", linewidth=1)

    # SAP
    s = 1
    sel_AP = SAP
    sel_AP_v = SAP_V

    ylim = 200
    xlim = 200

    ax[s].text(30, ylim - 35, "E", fontsize=12, ha="center", va="center", color="black")
    ax[s].text(84, ylim - 35, "D", fontsize=12, ha="center", va="center", color="black")
    ax[s].text(
        110, ylim - 35, "C", fontsize=12, ha="center", va="center", color="black"
    )
    ax[s].text(
        135, ylim - 35, "B", fontsize=12, ha="center", va="center", color="black"
    )
    ax[s].text(
        xlim - 20,
        ylim - 35,
        "A",
        fontsize=12,
        ha="center",
        va="center",
        color="black",
    )
    ax[s].text(xlim - 10, 30, "E", fontsize=12, ha="center", va="center", color="black")
    ax[s].text(xlim - 20, 80, "D", fontsize=12, ha="center", va="center", color="black")
    ax[s].text(
        xlim - 20, 110, "C", fontsize=12, ha="center", va="center", color="black"
    )
    ax[s].text(
        xlim - 20, 130, "B", fontsize=12, ha="center", va="center", color="black"
    )

    for key in sel_AP.keys():
        vertices = np.array(sel_AP[key])
        ax[s].plot(vertices[:, 0], vertices[:, 1], c=C_MAP[colormap][key[0]])
    for key in sel_AP_v.keys():
        vertices = sel_AP_v[key]
        polygon = Polygon(
            vertices,
            closed=True,
            edgecolor="black",
            facecolor=C_MAP[colormap][key[0]],
            alpha=0.4,
            zorder=1,
        )
        ax[s].add_patch(polygon)
    ax[s].set_xlim(0, 200)
    ax[s].set_ylim(0, 200)

    sel_AP = SAP
    sel_AP_v = SAP_V
    ax[s].set_title("SAP")
    ax[s].scatter(
        x=X[:, 1],
        y=Y[:, 1],
        s=dot_size,
        c="k",
    )
    sns.kdeplot(
        x=X[:, 1],
        y=Y[:, 1],
        cmap="rocket",
        fill=True,
        # clip=(-5, 5),
        # cut=10,
        # thresh=0,
        levels=levels,  # levels=[0.2, 0.4, 0.6,0.8,1],
        ax=ax[s],
    )
    ax[s].set_xlabel("Gold Standard (mmHg)")
    # MAP
    s = 2

    sel_AP = MAP
    sel_AP_v = MAP_V
    for key in sel_AP.keys():
        vertices = np.array(sel_AP[key])
        ax[s].plot(vertices[:, 0], vertices[:, 1], c=C_MAP[colormap][key[0]])
    for key in sel_AP_v.keys():
        vertices = sel_AP_v[key]
        polygon = Polygon(
            vertices,
            closed=True,
            edgecolor="black",
            facecolor=C_MAP[colormap][key[0]],
            alpha=0.4,
            zorder=1,
        )
        ax[s].add_patch(polygon)

    ax[s].set_title("MAP")
    ax[s].scatter(
        x=X[:, 2],
        y=Y[:, 2],
        s=dot_size,
        c="k",
    )
    sns.kdeplot(
        x=X[:, 2],
        y=Y[:, 2],
        cmap="rocket",
        fill=True,
        # clip=(-5, 5),
        # cut=10,
        # thresh=0,
        levels=levels,
        ax=ax[s],
    )
    ax[s].set_xlim(0, 200)
    ax[s].set_ylim(0, 200)
    ax[s].set_xlabel("Gold Standard (mmHg)")

    ylim = 200
    xlim = 200

    ax[s].text(30, ylim - 35, "E", fontsize=12, ha="center", va="center", color="black")
    ax[s].text(68, ylim - 35, "D", fontsize=12, ha="center", va="center", color="black")
    ax[s].text(85, ylim - 35, "C", fontsize=12, ha="center", va="center", color="black")
    ax[s].text(
        110, ylim - 35, "B", fontsize=12, ha="center", va="center", color="black"
    )
    ax[s].text(
        xlim - 20,
        ylim - 35,
        "A",
        fontsize=12,
        ha="center",
        va="center",
        color="black",
    )
    ax[s].text(xlim - 20, 30, "E", fontsize=12, ha="center", va="center", color="black")
    ax[s].text(xlim - 20, 70, "D", fontsize=12, ha="center", va="center", color="black")
    ax[s].text(xlim - 20, 92, "C", fontsize=12, ha="center", va="center", color="black")
    ax[s].text(
        xlim - 20, 110, "B", fontsize=12, ha="center", va="center", color="black"
    )

    plt.tight_layout()
    if sname != "none":
        plt.savefig(sname)
    plt.show()


def hist_AP(y_pred, y_true, AP):
    """AP = SAP or MAP"""
    hist_dict = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
    if AP == "MAP":
        sel_AP = MAP
        sel_AP_v = MAP_V
        s_idx = 2
    elif AP == "SAP":
        sel_AP = SAP
        sel_AP_v = SAP_V
        s_idx = 1
    for key in sel_AP_v.keys():
        tmp_poly = shape_geom.Polygon(sel_AP_v[key])
        for y, x in zip(y_pred[:, s_idx], y_true[:, s_idx]):
            tmp_p = shape_geom.Point(x, y)
            if tmp_p.within(tmp_poly):
                hist_dict[key[0]] += 1
    df_AP = pd.DataFrame.from_dict(hist_dict, orient="index", columns=["Count"])
    df_AP.reset_index(inplace=True)
    df_AP.rename(columns={"index": "Category"}, inplace=True)
    return df_AP


def plot_SAP_MAP(df_SAP, df_MAP, sname="none"):
    # set layout-theme
    sns.set_theme(style="white", context="talk")
    plt.rcParams.update({"font.size": 12})
    plt.rcParams["axes.grid"] = True
    sns.set(font_scale=1.2)

    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    sns.set_context(context="paper", font_scale=1.4)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    plt.subplot(1, 2, 1)
    sns.barplot(
        x="Category",
        y="Count",
        data=df_SAP,
        hue="Category",
        palette=C_MAP["python"],
        alpha=0.4,
    )
    for index, row in df_SAP.iterrows():
        plt.text(
            index,
            row["Count"] // 2 if row["Count"] > 20 else row["Count"],
            f"{row['Count']/df_SAP['Count'].sum()*100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=13,
        )
    plt.xlabel(f"SAP Category")
    plt.ylabel("Number of samples")

    plt.subplot(1, 2, 2)
    sns.barplot(
        x="Category",
        y="Count",
        data=df_MAP,
        hue="Category",
        palette=C_MAP["python"],
        alpha=0.4,
    )
    for index, row in df_MAP.iterrows():
        plt.text(
            index,
            row["Count"] // 2 if row["Count"] > 20 else row["Count"],
            f"{row['Count']/df_MAP['Count'].sum()*100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=13,
        )
    plt.xlabel(f"MAP Category")
    plt.ylabel("Number of samples")

    plt.tight_layout()
    if sname != "none":
        plt.savefig(sname)
    plt.show()
