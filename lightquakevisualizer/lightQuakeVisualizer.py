#!/usr/bin/env python3
import argparse
import vtk
from vtk.util import numpy_support
import numpy as np
import seissolxdmf
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import importlib

pv.global_theme.nan_color = "white"


class seissolxdmfExtended(seissolxdmf.seissolxdmf):
    def ComputeTimeIndices(self, at_time):
        """retrive list of time index in file"""
        outputTimes = np.array(super().ReadTimes())
        idsOutputTimes = list(range(0, len(outputTimes)))
        lidt = []
        for oTime in at_time:
            if not oTime.startswith("i"):
                idsClose = np.where(np.isclose(outputTimes, float(oTime), atol=0.0001))
                if not len(idsClose[0]):
                    print(f"t={oTime} not found in {super().xdmfFilename}")
                else:
                    lidt.append(idsClose[0][0])
            else:
                sslice = oTime[1:]
                if ":" in sslice or sslice == "-1":
                    parts = sslice.split(":")
                    startstopstep = [None for i in range(3)]
                    for i, part in enumerate(parts):
                        startstopstep[i] = int(part) if part else None
                    lidt.extend(
                        idsOutputTimes[
                            startstopstep[0] : startstopstep[1] : startstopstep[2]
                        ]
                    )
                else:
                    lidt.append(int(sslice))
        return sorted(list(set(lidt)))

    def ReadData(self, dataName, idt=-1):
        if dataName == "SR" and "SR" not in super().ReadAvailableDataFields():
            SRs = super().ReadData("SRs", idt)
            SRd = super().ReadData("SRd", idt)
            return np.sqrt(SRs**2 + SRd**2)
        if dataName == "rake" and "rake" not in super().ReadAvailableDataFields():
            Sls = super().ReadData("Sls", idt)
            Sld = super().ReadData("Sld", idt)
            ASl = super().ReadData("ASl", idt)
            # seissol has a unusual convention positive Sls for right-lateral, hence the -
            rake = np.degrees(np.arctan2(Sld, -Sls))
            rake[ASl < 0.01] = np.nan
            if np.nanpercentile(np.abs(rake), 90) > 150:
                rake[rake < 0] += 360
            # print(np.nanmin(rake), np.nanmax(rake))
            return rake
        else:
            return super().ReadData(dataName, idt)


def create_vtk_grid(
    xyz: np.ndarray, connect: np.ndarray
) -> vtk.vtkPolyData or vtk.vtkUnstructuredGrid:
    """Create a VTK grid from HDF5 data.

    Parameters:
    xyz (np.ndarray): Node coordinates
    connect (np.ndarray): Connectivity array

    Returns:
    vtk.vtkUnstructuredGrid: The created VTK grid
    """
    n_elements, ndim2 = connect.shape
    grid_type = {3: vtk.vtkPolyData, 4: vtk.vtkUnstructuredGrid}[ndim2]
    grid = grid_type()

    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(xyz))
    grid.SetPoints(points)

    cells = vtk.vtkCellArray()
    connect2 = np.zeros((n_elements, ndim2 + 1), dtype=np.int64)
    # number of points in the cell
    connect2[:, 0] = ndim2
    connect2[:, 1:] = connect
    cells.SetCells(n_elements, numpy_support.numpy_to_vtkIdTypeArray(connect2))

    if ndim2 == 3:
        grid.SetPolys(cells)
    else:
        grid.SetCells(vtk.VTK_TETRA, cells)

    return grid


def get_available_cmaps() -> dict:
    """
    Get a dictionary of available colormaps for each library.

    Returns:
    dict: A dictionary with library names as keys and lists of available colormaps as values.
    """
    avaiable_cmaps = {}
    avaiable_cmaps["matplotlib"] = plt.colormaps()
    cm = importlib.import_module("cmcrameri.cm")
    if cm:
        avaiable_cmaps["cmcrameri"] = cm.cmaps.keys()
    cm = importlib.import_module("cmasher")
    if cm:
        avaiable_cmaps["cmasher"] = cm.get_cmap_list()
    return avaiable_cmaps


def get_cmap(cmap_name: str, cmap_lib: str) -> object:
    """
    Get a colormap object from a library.

    Args:
    cmap_name (str): The name of the colormap.
    cmap_lib (str): The library to get the colormap from.

    Returns:
    object: The colormap object.

    Raises:
    ValueError: If the cmap_lib is unknown.
    """
    if cmap_lib == "matplotlib":
        from matplotlib import colormaps

        return colormaps[cmap_name]
    elif cmap_lib == "cmasher":
        import cmasher as cm

        return getattr(cm, cmap_name)
    elif cmap_lib == "cmcrameri":
        from cmcrameri import cm

        return cm.cmaps[cmap_name]
    else:
        raise ValueError(f"unkown cmap_lib {cmap_lib}")


def get_cmaps_objects(cmap_names: list) -> list:
    """
    Get a list of colormap objects from a list of colormap names.

    Args:
    cmap_names (list): A list of colormap names.
    if a colormap is a know colormap with prepended 0
    then the first color is changed to white

    Returns:
    list: A list of colormap objects.

    Raises:
    ValueError: If a colormap name is unknown.
    """
    cmaps_objects = []
    available_cmaps = get_available_cmaps()
    for cmap_name in cmap_names:
        if cmap_name[-1] == "0":
            change_to_white_first = True
            cmap_name_no0 = cmap_name[0:-1]
        else:
            change_to_white_first = False
            cmap_name_no0 = cmap_name
        found = False
        for cmaplib in available_cmaps.keys():
            if cmap_name_no0 in available_cmaps[cmaplib]:
                found = True
                # print(f"{cmap_name_no0} found in {cmaplib}")
                cmap = get_cmap(cmap_name_no0, cmaplib)
                if change_to_white_first:
                    num_colors = 256
                    # Create a new colormap that starts with white
                    white = np.array([1, 1, 1, 1])
                    new_colors = np.vstack(
                        (white, cmap(np.linspace(0, 1, num_colors - 1)))
                    )
                    # Create a new colormap object
                    cmap = mcolors.ListedColormap(
                        new_colors, name=cmap_name, N=num_colors
                    )
                cmaps_objects.append(cmap)
                break
        if not found:
            raise ValueError(f"unkown cmap: {cmap_name_no0}")
    return cmaps_objects


def add_contours(
    plotter: pv.Plotter,
    grid: vtk.vtkUnstructuredGrid,
    sx: seissolxdmfExtended,
    i: int,
    idt: int,
    args_contours: str,
) -> None:
    """
    Add contours to a plotter based on contour parameters.

    Args:
        plotter: A pyvista plotter object.
        grid: A vtk grid object.
        sx: a seissolxdmfExtended object to read seissol data
        i: An integer, indexing the output file.
        idt: An integer, indexing the time snapshot.
        args_contours: A string containing contour parameters, separated by semicolons.

    Returns:
        None

    Example:
        args_contours = "1 var1 3 red 2 0 10 1; 2 var2 2 blue 1 0 5 0.5"
        add_contours(plotter, grid, 1, 1, args_contours)
    """
    for contour_param in args_contours.split(";"):
        params = contour_param.split()
        idc, varc, number_contours = params[0:3]
        idc, number_contours = int(idc), int(number_contours)
        if idc != i:
            continue
        error_msg = "contour params should be: id variable nb_of_cont (color thickness min max dx)*nb_of_cont"
        assert len(params) == 5 * number_contours + 3, error_msg
        myData = sx.ReadData(varc, idt)
        vtkArray = numpy_support.numpy_to_vtk(
            num_array=myData, deep=True, array_type=vtk.VTK_FLOAT
        )
        vtkArray.SetName(varc)
        grid.GetCellData().AddArray(vtkArray)
        mesh = pv.wrap(grid)
        grid.GetCellData().RemoveArray(varc)
        print("using a threshold of 0.1 m for contour plots")
        mesh = mesh.threshold(value=(0.1, mesh["ASl"].max()), scalars="ASl")
        mesh = mesh.cell_data_to_point_data([varc])

        for k in range(number_contours):
            colorc = params[3 + 5 * k]
            thickc = float(params[4 + 5 * k])
            minc = params[5 + 5 * k]
            minc = myData.min() if minc == "min" else float(minc)
            maxc = params[6 + 5 * k]
            maxc = myData.max() if maxc == "max" else float(maxc)
            dxc = float(params[7 + 5 * k])
            print(
                f"generating contour for {varc}: np.arange({minc}, {maxc}, {dxc}), in {colorc} with line_width {thickc}"
            )
            contours = mesh.contour(np.arange(minc, maxc, dxc), scalars=varc)
            plotter.add_mesh(contours, color=colorc, line_width=thickc)


def configure_camera(plotter: pv.Plotter, mesh: pv.PolyData, view_arg: str) -> None:
    """
    Configure the camera for a PyVista plotter based on a view argument.

    Args:
        plotter: A PyVista plotter object.
        mesh: A PyVista mesh object.
        view_arg: A string specifying the view, either a file path to a.pvcc file or a predefined view name (xy, xz, yz, normal).

    Returns:
        None

    Notes:
        If view_arg is a.pvcc file, the camera is configured from the file.
        If view_arg is a predefined view name, the camera is configured accordingly.
        If view_arg is unknown, a ValueError is raised.

    Example:
        configure_camera(plotter, mesh, "xy")
        configure_camera(plotter, mesh, "path/to/view.pvcc")
    """
    view_name, view_ext = os.path.splitext(os.path.basename(view_arg))
    is_pvcc = view_ext == ".pvcc"

    match view_name:
        case "xy":
            plotter.view_xy()
        case "xz":
            plotter.view_xz()
        case "yz":
            plotter.view_yz()
        case "normal":
            center = mesh.center
            plane_normal = mesh.compute_normals()["Normals"].mean(axis=0)
            if plane_normal[2] < 0:
                plane_normal = -plane_normal
            plotter.camera.focal_point = center
            plotter.camera.position = center + plane_normal
        case _:
            if is_pvcc:
                plotter.camera = pv.Camera.from_paraview_pvcc(view_arg)
            else:
                raise ValueError(f"unknown view name {view_name}")

    plotter.parallel_scale = 1
    fp = plotter.camera.focal_point
    plotter.reset_camera()
    if is_pvcc:
        plotter.camera.focal_point = fp


def main():
    parser = argparse.ArgumentParser(
        description="Visualize SeisSol output using pyvista"
    )

    parser.add_argument(
        "input_files",
        help="SeisSol XDMF filename(s) to visualize, separated by ';'",
    )

    parser.add_argument(
        "--annotate_time",
        nargs=1,
        metavar="color xr yr",
        help="Display the time on the plot (xr and yr are relative location of the text)",
    )

    parser.add_argument(
        "--vtk_meshes",
        nargs=1,
        metavar="fname color linewidth",
        help="Plot VTK meshes (e.g. coastline), group of 3 arguments separated by ';'",
    )

    parser.add_argument(
        "--color_ranges",
        nargs=1,
        help="Color range for each file, separated by ';'",
    )

    parser.add_argument(
        "--contours",
        nargs=1,
        help=(
            "3 + 5*n parameters per contour_variable, with n number of contour:"
            "index of the file, variable, n, and for each contour "
            "color, line_width, min, max, dx of np.arange"
            ". Coutour parameters (group of 3 + 5n params) separated by ';'"
        ),
    )

    parser.add_argument(
        "--cmap", nargs=1, help="cmap for each file, separated by ';'", required=True
    )

    parser.add_argument(
        "--font_size",
        nargs=1,
        metavar="fs",
        help="Font-size of VTK objects",
        type=int,
        default=([20]),
    )

    parser.add_argument(
        "--hide_boundary_edges",
        dest="hide_boundary_edges",
        default=False,
        action="store_true",
        help="Hide boundary edges",
    )

    parser.add_argument(
        "--interactive",
        dest="interactive",
        action="store_true",
        help="Plots interactively (opens a popup)",
    )

    parser.add_argument(
        "--lighting",
        nargs=3,
        metavar=("specular", "diffuse", "ambient"),
        help="Lighting parameters",
        type=float,
        default=([0.1, 0.8, 0.5]),
    )

    parser.add_argument(
        "--opacity",
        nargs=1,
        help="Opacity values, separated by ';'",
    )

    parser.add_argument(
        "--output_prefix", nargs=1, help="Specify output prefix of the snapshot"
    )

    parser.add_argument(
        "--view",
        nargs=1,
        default=["normal"],
        metavar="pvcc_file_or_specific_view",
        help="Setup the camera view: e.g. normal, xy, xz, yz or path to a pvcc_file",
    )

    parser.add_argument(
        "--scalar_bar",
        nargs=1,
        metavar="xr yr (height_pxl)",
        help="Show scalar bar",
    )

    parser.add_argument(
        "--time",
        nargs=1,
        default=["i-1"],
        help=(
            "Simulation time or steps to vizualize, separated by ';'. prepend a i for a"
            " step, or a Python slice notation. E.g. 45.0;i2;i4:10:2;i-1 will extract a"
            " snapshot at simulation time 45.0, the 2nd time step, and time steps 4,6, 8"
            " and the last time step. If several files are vizualized simultaneously step"
            " and pythonslices options based on the first file"
        ),
    )

    parser.add_argument(
        "--variables",
        nargs=1,
        help="Variable(s) to visualize, separated by ';'",
        required=True,
    )

    parser.add_argument(
        "--window_size",
        nargs=2,
        metavar=("width", "height"),
        default=([1200, 900]),
        help="Size of the window, in pixels",
        type=int,
    )

    parser.add_argument(
        "--zoom", nargs=1, metavar="zoom", help="Camera zoom", type=float
    )

    args = parser.parse_args()

    if not os.path.exists("output") and not args.interactive:
        os.makedirs("output")

    fnames = args.input_files.split(";")
    variables = args.variables[0].split(";")
    cmap_names = args.cmap[0].split(";")
    nfiles = len(fnames)
    opacity = (
        [float(v) for v in args.opacity[0].split(";")]
        if args.opacity
        else np.ones(nfiles)
    )
    assert len(opacity) == nfiles

    def gen_color_range(scolor_ranges):
        color_ranges_pairs = scolor_ranges[0].split(";")
        color_ranges = []
        for cr_pairs in color_ranges_pairs:
            clim = [float(v) for v in cr_pairs.split()]
            color_ranges.append({"clim": clim})
        return color_ranges

    lighting = {
        "specular": args.lighting[0],
        "diffuse": args.lighting[1],
        "ambient": args.lighting[2],
    }
    if args.color_ranges:
        color_ranges = gen_color_range(args.color_ranges)
    dic_window_size = {"window_size": args.window_size}

    cmaps = get_cmaps_objects(cmap_names)

    def get_snapshot_fname(args, fname, mytime):
        if args.output_prefix:
            basename = args.output_prefix[0]
        else:
            mod_prefix = os.path.splitext(fname)[0].replace("/", "_")
            svar = args.variable[0].replace(";", "_")
            view_name, view_ext = os.path.splitext(os.path.basename(args.view[0]))
            is_pvcc = view_ext == ".pvcc"
            spvcc = f"_{view_name}_" if is_pvcc else ""
            basename = f"{mod_prefix}{spvcc}{svar}_{mytime}"
        return f"output/{basename}.png"

    sx = seissolxdmfExtended(fnames[0])
    time_indices = sx.ComputeTimeIndices(args.time[0].split(";"))
    outputTimes = sx.ReadTimes()
    filtered_list = []
    nOutputTimes = len(outputTimes)
    for x in time_indices:
        if -nOutputTimes <= x < nOutputTimes:
            filtered_list.append(x)
        else:
            print(f"Warning: time index {x} removed as out of range.")
    times = [outputTimes[k] for k in filtered_list]
    if not len(times):
        raise ValueError("all time index given are invalid")
    print(f"snapshots will be generated at times: {times}")

    def generate_snap(mytime):
        plotter = pv.Plotter(off_screen=not args.interactive, **dic_window_size)
        for i, fname in enumerate(fnames):
            sx = seissolxdmfExtended(fname)
            xyz = sx.ReadGeometry()
            connect = sx.ReadConnect()
            grid = create_vtk_grid(xyz, connect)
            var = variables[i]
            idx = sx.ComputeTimeIndices([str(mytime)])
            if len(idx) == 0:
                print(f"no output at t={mytime}s found for {fname}, skipping...")
                continue
            myData = sx.ReadData(var, idx[0])
            vtkArray = numpy_support.numpy_to_vtk(
                num_array=myData, deep=True, array_type=vtk.VTK_FLOAT
            )
            vtkArray.SetName(var)
            grid.GetCellData().SetScalars(vtkArray)
            mesh = pv.wrap(grid)
            clim_dic = color_ranges[i] if args.color_ranges else {"clim": None}

            if args.scalar_bar:
                sb_args = args.scalar_bar[0].split()
                height_pxl = int(sb_args[2]) if len(sb_args) == 3 else 150
                xr, yr = [float(v) for v in sb_args[0:2]]
                height = height_pxl / args.window_size[1]
                width = 40 / args.window_size[0]
                # shift successive scalar bars
                xr += 2 * width * i
                scalar_bar_dic = {
                    "scalar_bar_args": dict(
                        width=width,
                        height=height,
                        vertical=True,
                        position_x=xr,
                        position_y=yr,
                        label_font_size=int(1.8 * args.font_size[0]),
                        title_font_size=int(1.8 * args.font_size[0]),
                        n_labels=3,
                        fmt="%g",
                        title=var if var != "SR" else "slip rate (m/s)",
                    )
                }
            else:
                scalar_bar_dic = {}

            plotter.add_mesh(
                mesh,
                cmap=cmaps[i],
                scalars=var,
                **lighting,
                opacity=opacity[i],
                **clim_dic,
                **scalar_bar_dic,
            )

            if args.contours:
                add_contours(plotter, grid, sx, i, idx[0], args.contours[0])

            if not args.scalar_bar:
                plotter.remove_scalar_bar()
            if (not args.hide_boundary_edges) and ("surface" not in fname):
                edges = mesh.extract_feature_edges(
                    boundary_edges=True,
                    feature_edges=False,
                    manifold_edges=False,
                    non_manifold_edges=False,
                )
                plotter.add_mesh(edges, color="k", line_width=2)

        configure_camera(plotter, mesh, args.view[0])
        if args.vtk_meshes:
            list_vtk_mesh_args = args.vtk_meshes[0].split(";")
            for vtk_mesh_args in list_vtk_mesh_args:
                fname, color, line_width = vtk_mesh_args.split()
                vtk_mesh = pv.read(fname)
                plotter.add_mesh(vtk_mesh, color=color, line_width=int(line_width))

        if args.zoom:
            plotter.camera.zoom(args.zoom[0])

        if args.annotate_time:
            colname, xr, yr = args.annotate_time[0].split()
            x1 = float(xr) * args.window_size[0]
            y1 = float(yr) * args.window_size[1]

            plotter.add_text(
                f"{mytime:.1f}s",
                position=(x1, y1),
                color=colname,
                font_size=args.font_size[0],
            )

        if args.interactive:
            plotter.show()
        else:
            out_fname = get_snapshot_fname(args, fname, mytime)
            plotter.screenshot(out_fname)
            print(f"done writing {out_fname}")
        plotter.close()
        plotter.deep_clean()

    for mytime in times:
        generate_snap(mytime)


if __name__ == "__main__":
    main()
