import argparse
from collections.abc import Hashable, Iterable, Sequence
from functools import reduce
import os
import pathlib
import re
import sys
from typing import Any, Mapping, Optional, TypeVar, TypeVarTuple, overload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

T = TypeVar("T")
S = TypeVar("S")
Args = TypeVarTuple("Args")

KT = TypeVar("KT")
VT1 = TypeVar("VT1")
VT2 = TypeVar("VT2")


def dict_zip(a: Mapping[KT, VT1], b: Mapping[KT, VT2]) -> Iterable[tuple[KT, VT1, VT2]]:
    return iter((k, v, b[k]) for k, v in a.items())


@overload
def groupby_level(
    df: pd.DataFrame, level: Hashable
) -> Mapping[Hashable, pd.DataFrame]: ...


@overload
def groupby_level(df: pd.Series, level: Hashable) -> Mapping[Hashable, pd.Series]: ...


def groupby_level(
    df: pd.DataFrame | pd.Series, level: Hashable
) -> Mapping[Hashable, pd.DataFrame | pd.Series]:
    rval = dict(tuple(df.groupby(level=level)))
    for value in rval.values():
        value.index = value.index.droplevel(level=level)
    return rval


def read_data(fname: os.PathLike) -> pd.DataFrame:
    def converter(x: str | float) -> float:
        if isinstance(x, float | int):
            return x
        return float(x.removesuffix("% ").replace(",", ""))

    data = (
        pd.read_table(
            fname,
            sep="|",
            skipinitialspace=True,
            header=0,
            skiprows=[1],
            index_col=-1,
        )
        .iloc[:, 1:]
        .map(converter)
    )

    pattern = re.compile(
        r"`(Energy|Update): Geometry\( ([0-9]+)x([0-9]+)x([0-9]+), cell: ([0-9]+) \)`"
    )
    types = (str, int, int, int, int)

    def matcher(arg: str):
        match = pattern.search(arg)
        if match is None:
            print(pattern)
            print(arg)
            raise RuntimeError
        return tuple(t(item) for t, item in zip(types, match.groups()))

    data.index = pd.MultiIndex.from_tuples(
        [matcher(x) for x in data.index], names=("label", "n1", "n2", "n3", "n0")
    )
    data.columns = [label.strip() for label in data.columns]
    data["ncells"] = [
        reduce(lambda x, y: x * y, row)
        for row in data.index.droplevel("label").droplevel("n0")
    ]

    return data


def minmax(df: pd.DataFrame, level="label") -> Mapping[Hashable, pd.DataFrame]:
    dfmin = df.groupby(level=level).min()
    dfmax = df.groupby(level=level).max()
    rval = pd.concat([dfmin.unstack(), dfmax.unstack()], axis=1)
    rval.columns = ["min", "max"]
    return groupby_level(rval, level)


def common_minmax(
    A: pd.DataFrame, B: pd.DataFrame
) -> pd.DataFrame:
    rval = pd.concat(
        [
            pd.concat([A["min"], B["min"]], axis=1).min(axis=1),
            pd.concat([A["max"], B["max"]], axis=1).max(axis=1),
        ],
        axis=1,
    )
    rval.columns = ["min", "max"]
    return rval


def plot(
    data: pd.DataFrame,
    basename: str,
    plot_dir: pathlib.Path = pathlib.Path("."),
    limits: Optional[Mapping[Hashable, pd.DataFrame]] = None,
) -> None:
    runs = groupby_level(data, "label")
    s = 3.2
    wspace = 0.2
    nrows = len(runs)
    ncols = 4

    row_pad = 1
    col_pad = 5
    text_kwargs = dict(fontsize="large")

    def kw_col_header(_) -> dict:
        return dict(
            xy=(0.5, 1),
            xytext=(0, col_pad),
            xycoords="axes fraction",
            textcoords="offset points",
            ha="center",
            va="baseline",
            **text_kwargs,
        )

    def kw_row_header(ax) -> dict:
        return dict(
            xy=(0, 0.5),
            xytext=(-ax.yaxis.labelpad - row_pad, 0),
            xycoords=ax.yaxis.label,
            textcoords="offset points",
            ha="right",
            va="center",
            rotation=90,
            **text_kwargs,
        )

    def annotate(ax, col_header=None, row_header=None, xlabel=None, ylabel=None):
        sps = ax.get_subplotspec()
        if (col_header is not None) and sps.is_first_row():
            ax.annotate(col_header, **kw_col_header(ax))
        if (row_header is not None) and sps.is_first_col():
            ax.annotate(row_header, **kw_row_header(ax))
        if (xlabel is not None) and sps.is_last_row():
            ax.set_xlabel(xlabel)
        if (ylabel is not None) and sps.is_first_col():
            ax.set_ylabel(ylabel)

    def set_limits(f, vmin, vmax, pad=0.1):
        pad = 0.1 * (vmax - vmin)
        if pad > 0:
            f(vmin - pad, vmax + pad)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        dpi=300,
        squeeze=False,
        figsize=((ncols + wspace * (ncols - 1)) * s, nrows * s),
        gridspec_kw=dict(wspace=wspace, left=0.07, right=0.95),
    )

    for axrow, (label, datagroup) in zip(axes, runs.items()):

        sets = groupby_level(datagroup, "n0")
        # execution time
        ax = axrow[0]
        cidx = "ns/op"
        for n0, df in sets.items():
            xdata = df["ncells"].to_numpy()
            ydata = df[cidx].to_numpy()
            yerr = (df[cidx] * df["err%"]).to_numpy()
            ax.errorbar(
                xdata,
                ydata,
                yerr=yerr,
                fmt="o",
                linewidth=2,
                capsize=3,
                label=n0,
            )

            # if fit_function is not None:
            #     res = curve_fit(fit_function, xdata, ydata, sigma=yerr)
            #     fitx = np.linspace(xdata.min(), xdata.max(), int(1e5))
            #     fity = fit_function(xdata, *res.popt)
            #     ax.plot(fitx, fity, label=None)

        ax.set_ylabel(cidx)
        ax.legend()
        annotate(ax, row_header=label, col_header="Execution Time", xlabel="# cells")
        if not ax.get_subplotspec().is_first_row():
            ax.sharex(axes[0, 0])
        if limits is not None:
            set_limits(ax.set_ylim, *limits[label].loc[cidx, :])

        # instructions per cycle
        ax = axrow[1]
        cidx = "IPC"
        hist_data = datagroup[cidx].unstack("n0")
        ax.hist(hist_data, histtype="bar", stacked=True, label=hist_data.columns)
        ax.legend()
        annotate(
            ax,
            row_header=label,
            col_header="Instructions per Cycle",
            xlabel=f"# {cidx}",
        )
        if not ax.get_subplotspec().is_first_row():
            ax.sharex(axes[0, 1])
        elif limits is not None:
            vmin = reduce(min, (v.loc[cidx, "min"] for v in limits.values()))
            vmax = reduce(max, (v.loc[cidx, "max"] for v in limits.values()))
            set_limits(ax.set_xlim, vmin, vmax)

        # branches per operation
        ax = axrow[2]
        cidx = "bra/op"
        for n0, df in sets.items():
            ax.scatter(df["ncells"], df[cidx], label=n0)
        ax.set_ylabel(cidx)
        ax.legend()
        annotate(
            ax, row_header=label, col_header="Branches per Operation", xlabel="# cells"
        )
        if not ax.get_subplotspec().is_first_row():
            ax.sharex(axes[0, 2])
        if limits is not None:
            set_limits(ax.set_ylim, *limits[label].loc[cidx, :])

        # miss%
        ax = axrow[3]
        cidx = "miss%"
        hist_data = datagroup[cidx].unstack("n0")
        ax.hist(hist_data, histtype="bar", stacked=True, label=hist_data.columns)
        ax.legend()
        annotate(ax, row_header=label, col_header="Branch miss rate", xlabel=cidx)
        if not ax.get_subplotspec().is_first_row():
            ax.sharex(axes[0, 3])
        elif limits is not None:
            vmin = reduce(min, (v.loc[cidx, "min"] for v in limits.values()))
            vmax = reduce(max, (v.loc[cidx, "max"] for v in limits.values()))
            set_limits(ax.set_xlim, vmin, vmax)
        else:
            vmin = min(0, np.nanmin(hist_data.to_numpy()))
            vmax = np.nanmax(hist_data.to_numpy())
            set_limits(ax.set_xlim, vmin, vmax)

    fig.suptitle(f"Benchmark: {basename}", fontsize="x-large")
    fig.savefig(plot_dir / f"{basename}.png")


def parse_args(argv: Sequence[str]) -> Any:
    parser = argparse.ArgumentParser("plot")
    parser.add_argument("fname", type=pathlib.Path, nargs="+")
    return parser.parse_args(argv)


def main():
    args = parse_args(sys.argv[1:])

    plot_dir = pathlib.Path("./plots")
    os.makedirs(plot_dir, mode=0o755, exist_ok=True)

    def bound_common_minmax(
        d1: Mapping[Hashable, pd.DataFrame], d2: Mapping[Hashable, pd.DataFrame]
    ) -> Mapping[Hashable, pd.DataFrame]:
        return {k: common_minmax(v1, v2) for k, v1, v2 in dict_zip(d1, d2)}

    data_limits = reduce(
        bound_common_minmax, (minmax(read_data(fname)) for fname in args.fname)
    )

    for fname in args.fname:
        basename = os.path.splitext(os.path.basename(fname))[0]
        data = read_data(fname)

        plot(data, basename, plot_dir, limits=data_limits)


if __name__ == "__main__":
    main()
