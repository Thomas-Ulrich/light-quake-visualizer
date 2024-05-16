#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.colors as colors
from matplotlib import colormaps
from math import log10
import argparse


def reverse_colourmap(
    cmap: LinearSegmentedColormap, name: str = "my_cmap_r"
) -> LinearSegmentedColormap:
    """Reverse a colormap."""
    reverse = []
    k = []

    for key in cmap._segmentdata:
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []

        for t in channel:
            data.append((1 - t[0], t[2], t[1]))
        reverse.append(sorted(data))

    LinearL = dict(zip(k, reverse))
    my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL)
    return my_cmap_r


def CreateDicFromJson(fname: str) -> dict:
    """Create a dictionary from a JSON file."""
    import json

    with open(fname) as fid:
        datastore = json.load(fid)[0]
    RGB = np.array(datastore["RGBPoints"])
    nd = len(RGB)
    RGB = RGB.reshape(nd // 4, 4)
    cdict = {}

    temp = np.zeros((nd // 4, 3))
    temp[:, 0] = (RGB[:, 0] - RGB[0, 0]) / (RGB[-1, 0] - RGB[0, 0])
    temp[:, 1] = RGB[:, 1]
    temp[:, 2] = temp[:, 1]
    cdict["red"] = tuple(map(tuple, temp))
    temp[:, 1] = RGB[:, 2]
    temp[:, 2] = temp[:, 1]
    cdict["green"] = tuple(map(tuple, temp))
    temp[:, 1] = RGB[:, 3]
    temp[:, 2] = temp[:, 1]
    cdict["blue"] = tuple(map(tuple, temp))
    print(cdict)
    return cdict


def main():
    parser = argparse.ArgumentParser(
        description="Plot a colorbar only (and allow a vector format for the colorbar)"
    )
    parser.add_argument("cmap", help="colormap name (e.g. magma_r)")
    parser.add_argument(
        "--labelfontsize",
        nargs=1,
        metavar=("labelfontsize"),
        default=[25],
        help="font size label",
        type=float,
    )
    parser.add_argument(
        "--nticks",
        nargs=1,
        metavar=("nticks"),
        default=[5],
        help="number of ticks",
        type=int,
    )
    parser.add_argument(
        "--extension",
        nargs=1,
        metavar=("ext"),
        default=(["svg"]),
        help="extension of output file",
    )
    parser.add_argument(
        "--drange",
        nargs=2,
        metavar=("dmin", "dmax"),
        default=([0, 0]),
        help="data range if differ from crange",
        type=float,
    )
    parser.add_argument(
        "--crange",
        nargs=2,
        metavar=("cmin", "cmax"),
        default=([0, 1]),
        help="color range",
        type=float,
    )
    parser.add_argument(
        "--heightAR",
        nargs=2,
        metavar=("height", "AR"),
        default=([5, 3.6]),
        help="height of figure and Aspect Ratio",
        type=float,
    )
    parser.add_argument(
        "--noMinor", dest="noMinor", action="store_true", help="do not show minor ticks"
    )
    parser.add_argument(
        "--horizontal",
        dest="horizontal",
        action="store_true",
        help="plot horizontal colorbar",
    )
    parser.add_argument(
        "--reverse", dest="reverse", action="store_true", help="reverse color map"
    )
    parser.add_argument(
        "--log",
        dest="log",
        action="store_true",
        help="log format (nticks does not apply)",
    )
    args = parser.parse_args()

    if (args.drange[1] - args.drange[0]) > 0:
        a = np.array([args.drange])
    else:
        a = np.array([args.crange])

    if args.horizontal:
        plt.figure(figsize=(args.heightAR[0], args.heightAR[0] / args.heightAR[1]))
    else:
        plt.figure(figsize=(args.heightAR[0] / args.heightAR[1], args.heightAR[0]))

    # everything is tweaked up for h=6
    fact = args.heightAR[0] / 5.0

    if args.cmap.endswith(".json"):
        cdict = CreateDicFromJson(args.cmap)
        custom = mpl.colors.LinearSegmentedColormap("custom", cdict)
        if args.reverse:
            custom = reverse_colourmap(custom)
    elif args.cmap.endswith(".txt"):
        cm_data = np.loadtxt(args.cmap)
        custom = LinearSegmentedColormap.from_list("broc", cm_data)
        if args.reverse:
            custom = reverse_colourmap(custom)
    elif args.cmap in colormaps:
        custom = colormaps[args.cmap]
        print(f"importing {args.cmap} from matplotlib")
    else:
        try:
            from cmcrameri import cm

            custom = getattr(cm, args.cmap)
            print(f"importing {args.cmap} from cmcrameri")
        except (ModuleNotFoundError, AttributeError):
            raise ValueError("unknown colormap")
    if args.reverse and not args.cmap.endswith(".json"):
        print("reverse works only with json color maps")

    if args.log:
        img = plt.imshow(a, cmap=custom, norm=colors.LogNorm())
    else:
        img = plt.imshow(a, cmap=custom)

    plt.gca().set_visible(False)

    if args.horizontal:
        cax = plt.axes([0.05, 0.35, 0.9, 0.15])
        orient = "horizontal"
    else:
        cax = plt.axes([0.05, 0.05, 0.15, 0.9])
        orient = "vertical"

    if args.log:
        cb = plt.colorbar(cax=cax, orientation=orient)
        args.noMinor = True
    else:
        cb = plt.colorbar(
            cax=cax,
            ticks=np.linspace(args.crange[0], args.crange[1], args.nticks[0]),
            orientation=orient,
        )
    cb.ax.tick_params(
        which="major",
        labelsize=args.labelfontsize[0],
        length=10 * fact,
        width=2 * fact,
        direction="inout",
    )
    cb.outline.set_linewidth(2 * fact)
    cb.ax.tick_params(which="minor", length=6 * fact, width=1.0 * fact)

    if not args.noMinor:
        cb.ax.minorticks_on()
    # cb.ax.yaxis.set_ticks([0,0.25,0.5,0.75,1,1.5], minor=False)
    # pl.show()
    if args.log:
        args.crange[1] = log10(args.crange[1])
    fn = f"colorbar{args.cmap}{args.crange[1]:.1f}.{args.extension[0]}"

    plt.savefig(fn, transparent=True)
    print(f"done writing {fn}")


if __name__ == "__main__":
    main()
