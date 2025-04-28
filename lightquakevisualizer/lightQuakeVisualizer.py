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
import h5py
from typing import List
from importlib.metadata import version

pv.global_theme.nan_color = "white"


def compute_time_indices(output_times: list, at_times: list[str]) -> list[int]:
    """Retrieve list of time indices in output_times that match the given string.

    Args:
        output_times: List of available time stamps.
        at_times: List of times to search for in the file. Times can be specified
            as floats or as indices prefixed with "i" (e.g. "i10" for the 10th time
            step).

    Returns:
        List of time indices that match the given times.
    """
    output_time_indices = list(range(0, len(output_times)))
    time_indices = set()
    for at_time in at_times:
        if not at_time.startswith("i"):
            close_indices = np.where(
                np.isclose(output_times, float(at_time), atol=0.0001)
            )[0]
            if close_indices.size > 0:
                time_indices.add(close_indices[0])
            else:
                print(f"Time {at_time} not found")
        else:
            sslice = at_time[1:]
            if ":" in sslice or int(sslice) < 0:
                parts = sslice.split(":")
                start_stop_step = [None for i in range(3)]
                for i, part in enumerate(parts):
                    start_stop_step[i] = int(part) if part else None
                time_indices.update(
                    output_time_indices[
                        start_stop_step[0] : start_stop_step[1] : start_stop_step[2]
                    ]
                )
            else:
                time_indices.add(int(sslice))
    return sorted(list(time_indices))


class seissolxdmfExtended(seissolxdmf.seissolxdmf):
    def ComputeTimeIndices(self, at_times: list[str]) -> list[int]:
        """Retrieve list of time indices in file that match the given string.

        Args:
            at_times: List of times to search for in the file. Times can be specified
                as floats or as indices prefixed with "i" (e.g. "i10" for the 10th
                time step).

        Returns:
            List of time indices that match the given times.
        """
        output_times = np.array(super().ReadTimes())
        return compute_time_indices(output_times, at_times)

    def ReadData(self, data_name: str, idt: int = -1) -> np.ndarray:
        """Read data from a SeisSol file and may compute and return a derived quantity.

        Args:
            data_name: Name of the data field to read.
            idt: Time index to read data from. Defaults to -1, which reads from the
                last time step.

        Returns:
            Numpy array containing the read data.

        Notes:
            If the data field is not available, this method may compute and return
            a derived quantity. For example, if "SR" is not available, it will be
            computed as sqrt(SRs**2 + SRd**2). Similarly, if "rake" is not available,
            it will be computed from Sls, Sld, and ASl.
        """
        available_datasets = super().ReadAvailableDataFields()
        if data_name == "SR" and "SR" not in available_datasets:
            SRs = super().ReadData("SRs", idt)
            SRd = super().ReadData("SRd", idt)
            return np.sqrt(SRs**2 + SRd**2)
        if data_name == "Vr_kms" and "Vr_kms" not in available_datasets:
            return super().ReadData("Vr", idt) / 1e3
        if (
            data_name == "shear_stress_MPa"
            and "shear_stress_MPa" not in available_datasets
        ):
            Td0 = super().ReadData("T_d", idt)
            Ts0 = super().ReadData("T_s", idt)
            return np.sqrt(Ts0**2 + Td0**2) / 1e6
        if (
            data_name == "shear_stress0_MPa"
            and "shear_stress0_MPa" not in available_datasets
        ):
            Td0 = super().ReadData("Td0", idt)
            Ts0 = super().ReadData("Ts0", idt)
            return np.sqrt(Ts0**2 + Td0**2) / 1e6
        if data_name == "rake" and "rake" not in available_datasets:
            Sls = super().ReadData("Sls", idt)
            Sld = super().ReadData("Sld", idt)
            ASl = super().ReadData("ASl", idt)
            # seissol has a unusual convention
            # positive Sls for right-lateral, hence the -
            rake = np.degrees(np.arctan2(Sld, -Sls))
            rake[ASl < 0.01] = np.nan
            if np.nanpercentile(np.abs(rake), 90) > 150:
                rake[rake < 0] += 360
            # print(np.nanmin(rake), np.nanmax(rake))
            return rake
        if data_name == "SCU" and "SCU" not in super().ReadAvailableDataFields():
            Td0 = super().ReadData("Td0", idt)
            Ts0 = super().ReadData("Ts0", idt)
            T0 = np.sqrt(Ts0**2 + Td0**2)
            Pn0 = super().ReadData("Pn0", idt)
            Mus = super().ReadData("Mud", 0)
            return T0 / (np.multiply(np.abs(Pn0), Mus))
        else:
            return super().ReadData(data_name, idt)


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
    dict: A dictionary with library names as keys and lists of
    available colormaps as values.
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


def parse_contour_args(args_contours: str) -> list:
    """Parse a human-readable contours argument string into a structured format."""
    contours_list = []
    current_entry = {}

    try:
        for line in args_contours.split():
            key, value = line.split("=", 1)

            if key == "file_index":
                if current_entry:  # Save previous entry before starting a new one
                    contours_list.append(current_entry)
                current_entry = {"file_index": int(value), "contours": []}

            elif key == "var":
                current_entry["variable"] = value

            elif key == "contour":
                color, thickness, min_val, max_val, dx = value.split(",")
                current_entry["contours"].append(
                    {
                        "color": color,
                        "line_width": float(thickness),
                        "min": min_val if min_val == "min" else float(min_val),
                        "max": max_val if max_val == "max" else float(max_val),
                        "dx": float(dx),
                    }
                )

        if current_entry:  # Add the last parsed entry
            contours_list.append(current_entry)

    except ValueError:
        message = f"could not read args_contours:'{args_contours}', \nexample of \
expected format: 'file_index=0 var=RT contour=grey,2,0,max,1 contour=black,4,0,max,5'\n\
each contour entry follows the following pattern contour=color,thickness,min,max,dx"
        raise ValueError(message)

    return contours_list


def add_contours(
    plotter: pv.Plotter,
    grid: vtk.vtkUnstructuredGrid,
    sx,
    i: int,
    idt: int,
    args_contours: str,
) -> None:
    """
    Add contours to a plotter based on contour parameters.

    Args:
        plotter: A pyvista plotter object.
        grid: A vtk grid object.
        sx: A seissolxdmfExtended object to read seissol data.
        i: An integer, indexing the output file.
        idt: An integer, indexing the time snapshot.
        args_contours: A structured string containing contour parameters.

    Returns:
        None

    Example:
        args_contours =
        "file_index=0 var=RT contour=grey,2,0,max,1 contour=black,4,0,max,5"
        add_contours(plotter, grid, 0, 1, args_contours)
    """
    contours_list = parse_contour_args(args_contours)

    for entry in contours_list:
        if entry["file_index"] != i:
            continue

        varc = entry["variable"]
        myData = sx.ReadData(varc, idt)

        vtkArray = numpy_support.numpy_to_vtk(
            num_array=myData, deep=True, array_type=vtk.VTK_FLOAT
        )
        vtkArray.SetName(varc)
        grid.GetCellData().AddArray(vtkArray)

        mesh = pv.wrap(grid)
        grid.GetCellData().RemoveArray(varc)

        print("Using a threshold of 0.1 m for contour plots")
        mesh = mesh.threshold(value=(0.1, mesh["ASl"].max()), scalars="ASl")
        mesh = mesh.cell_data_to_point_data([varc])

        for contour in entry["contours"]:
            colorc = contour["color"]
            thickc = contour["line_width"]
            minc = myData.min() if contour["min"] == "min" else float(contour["min"])
            maxc = myData.max() if contour["max"] == "max" else float(contour["max"])
            dxc = float(contour["dx"])

            print(
                f"Generating contour for {varc}: np.arange({minc}, {maxc}, {dxc}), "
                f"in {colorc} with line_width {thickc}"
            )
            contours = mesh.contour(np.arange(minc, maxc, dxc), scalars=varc)
            plotter.add_mesh(contours, color=colorc, line_width=thickc)


def compute_plane_normal_surface(mesh):
    mesh_normals = mesh.compute_normals()
    return mesh_normals.point_data["Normals"].mean(axis=0)


def compute_plane_normal(mesh):
    """Computes the plane normal from a PyVista mesh, handling UnstructuredGrid
    without normals."""
    # typically tandem output
    if isinstance(mesh, pv.MultiBlock):
        normals = []
        for block in mesh:
            normals.append(compute_plane_normal(block))
        return np.mean(np.array(normals), axis=0)
    # seissol volume output
    elif isinstance(mesh, pv.UnstructuredGrid):
        surface = mesh.extract_surface()
        return compute_plane_normal_surface(surface)
    # seissol surface output
    elif isinstance(mesh, pv.PolyData):
        return compute_plane_normal_surface(mesh)
    else:
        raise ValueError("Unsupported mesh type.")


def configure_camera(plotter: pv.Plotter, mesh: pv.PolyData, view_arg: str) -> None:
    """
    Configure the camera for a PyVista plotter based on a view argument.

    Args:
        plotter: A PyVista plotter object.
        mesh: A PyVista mesh object.
        view_arg: A string specifying the view, either a file path to a.pvcc file
                  or a predefined view name (xy, xz, yz, normal, normal-flip).

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
        case "normal" | "normal-flip":
            center = mesh.center
            try:
                plane_normal = compute_plane_normal(mesh)
                if view_name == "normal-flip":
                    plane_normal = -plane_normal

                plotter.camera.focal_point = center
                plotter.camera.position = center + plane_normal
            except AttributeError:
                print("cannot compute normal, using xy view instead")
                plotter.view_xy()
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


def format_time(t):
    """
    Converts time in seconds to "years y days d hours h minutes m seconds s" format.

    Args:
        t (float): Time in seconds.

    Returns:
        str: Formatted time string.
    """
    years = int(t / (60.0 * 60.0 * 24.0 * 365.25))
    t -= years * 60.0 * 60.0 * 24.0 * 365.25

    days = int(t / (60.0 * 60.0 * 24.0))
    t -= days * 60.0 * 60.0 * 24.0

    hours = int(t / (60.0 * 60.0))
    t -= hours * 60.0 * 60.0

    minutes = int(t / 60.0)
    seconds = t - minutes * 60.0

    formatted_time = ""
    if years > 0:
        formatted_time += f"{years}y "
    if days > 0:
        formatted_time += f"{days}d "
    if hours > 0:
        formatted_time += f"{hours}h "
    if minutes > 0:
        formatted_time += f"{minutes}m "
    if seconds > 0 or not formatted_time:
        formatted_time += f"{seconds:.1f}s"

    return formatted_time


def validate_parameter_count(
    parameter_list: List, parameter_description: str, expected_number: int
) -> None:
    """
    Verify that the number of parameters matches the expected count.

    Parameters:
    parameter_list (list): List of parameters to be checked.
    parameter_description (str): Description of the parameter type
    for error message formatting.
    expected_number (int): The expected number of parameters.

    Raises:
    ValueError: If the number of parameters does not match the expected count.
    """
    n_param = len(parameter_list)
    if n_param != expected_number:
        raise ValueError(
            f"{n_param} {parameter_description} given ({parameter_list}), \
            but {expected_number} expected"
        )


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
        type=str,
        metavar="color xr yr",
        help="Display the time on the plot (xr and yr are relative location)",
    )

    parser.add_argument(
        "--annotate_text",
        type=str,
        metavar="color xr yr text",
        help=(
            "Display custom annotation on the plot (xr and yr are relative locations)."
            " For several annotations, use multiple 'color xr yr text', ';'-separated"
        ),
    )

    parser.add_argument(
        "--color_ranges",
        type=str,
        help="Color range for each file, separated by ';'",
    )

    parser.add_argument(
        "--contours",
        type=str,
        help=(
            "Contour configuration in a structured format. Example:\n"
            "'file_index=0 var=RT contour=grey,2,0,max,1 contour=black,4,0,max,5'\n\n"
            "Each entry consists of:\n"
            "- 'file_index=N' (index of the file)\n"
            "- 'var=VAR_NAME' (variable to contour)\n"
            "- 'contour=color,thickness,min,max,dx' (one per contour level)\n"
            "  - color (e.g., grey, black)\n"
            "  - thickness (line width)\n"
            "  - min (min value, can be 'min' for auto)\n"
            "  - max (max value, can be 'max' for auto)\n"
            "  - dx (contour step size)"
        ),
    )

    parser.add_argument(
        "--cmap", type=str, help="cmap for each file, separated by ';'", required=True
    )

    parser.add_argument(
        "--font_size",
        metavar="fs",
        help="Font-size of VTK objects",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--hide_boundary_edges",
        dest="hide_boundary_edges",
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
        default=[0.1, 0.8, 0.5],
    )

    parser.add_argument(
        "--log_scale",
        help="Log color scale. 1: log scale, 0: linear scale. n values ';' separated",
    )

    parser.add_argument(
        "--opacity",
        type=str,
        help="Opacity values, separated by ';'",
    )

    parser.add_argument(
        "--output_prefix",
        type=str,
        help=(
            "Specify output prefix of the snapshot, "
            "%%d will be replaced by the time index, "
            "{t:.2f} will be replaced by the time_value in the given format specifier"
        ),
    )

    parser.add_argument(
        "--scalar_bar",
        type=str,
        metavar="xr yr (height_pxl)",
        help="Show scalar bar",
    )

    parser.add_argument(
        "--slice",
        nargs=2,
        metavar=(
            "1st argument: slice plane defined by point and normal "
            "(x,y,z,nx,ny,nz), example 0 0 -2000 0 0 1. "
            "2nd argument: 1 or 0 for enabling or not slicing on "
            "given input file separated by ';'."
        ),
        help="slice outputs along plane",
    )

    parser.add_argument(
        "--time",
        default="i-1",
        type=str,
        help=(
            "Simulation time or steps to visualize, separated by ';'. prepend a i for "
            "a step, or a Python slice notation. e.g. 45.0;i2;i4:10:2;i-1 will extract "
            "a snapshot at simulation time 45.0, the 2nd time step, and time steps "
            "4,6,8, and the last time step. If several files are visualized "
            "simultaneously, step and Python slice options are based on the first file."
        ),
    )
    parser.add_argument(
        "--variables",
        type=str,
        help="Variable(s) to visualize, separated by ';'",
        required=True,
    )
    parser.add_argument(
        "--version", action="version", version=f'{version("lightquakevisualizer")}'
    )

    parser.add_argument(
        "--view",
        type=str,
        default=["normal"],
        metavar="pvcc_file_or_specific_view",
        help=(
            "Setup the camera view: e.g. "
            "normal, normal-flip, xy, xz, yz or path to a pvcc_file"
        ),
    )

    parser.add_argument(
        "--vtk_meshes",
        type=str,
        metavar="fname color linewidth",
        help="Plot VTK meshes (e.g. coastline), group of 3 arguments separated by ';'",
    )

    parser.add_argument(
        "--window_size",
        nargs=2,
        metavar=("width", "height"),
        default=[1200, 900],
        help="Size of the window, in pixels",
        type=int,
    )

    parser.add_argument("--zoom", metavar="zoom", help="Camera zoom", type=float)

    args = parser.parse_args()
    print(args)

    if not os.path.exists("output") and not args.interactive:
        os.makedirs("output")

    fnames = args.input_files.split(";")
    nfiles = len(fnames)

    variables = args.variables.split(";")
    validate_parameter_count(variables, "variables", nfiles)

    cmap_names = args.cmap.split(";")
    validate_parameter_count(cmap_names, "cmaps", nfiles)

    use_log_scale = (
        [True if int(v) else False for v in args.log_scale.split(";")]
        if args.log_scale
        else np.zeros(nfiles, dtype=bool)
    )
    validate_parameter_count(use_log_scale, "parameters in args.log_scale", nfiles)

    opacity = (
        [float(v) for v in args.opacity.split(";")] if args.opacity else np.ones(nfiles)
    )
    validate_parameter_count(opacity, "parameters in args.opacity", nfiles)

    def gen_color_range(scolor_ranges):
        color_ranges_pairs = scolor_ranges.split(";")
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

    def get_snapshot_fname(args, fname, itime, time_value):
        if args.output_prefix:
            basename = args.output_prefix.replace("%d", f"_{itime}")
            if "{t" in basename:
                basename = basename.format(t=float(time_value))

        else:
            mod_prefix = os.path.splitext(fname)[0].replace("/", "_")
            svar = args.variables.replace(";", "_")
            view_name, view_ext = os.path.splitext(os.path.basename(args.view))
            is_pvcc = view_ext == ".pvcc"
            spvcc = f"_{view_name}_" if is_pvcc else ""
            basename = f"{mod_prefix}{spvcc}{svar}_{itime}"
        return f"output/{basename}.png"

    if fnames[0].endswith("xdmf"):
        sx = seissolxdmfExtended(fnames[0])
        time_indices = sx.ComputeTimeIndices(args.time.split(";"))
        output_times = sx.ReadTimes()
    elif fnames[0].endswith("hdf"):
        print("reading a hdf file, no time information available")
        with h5py.File(fnames[0], "r") as f:
            output_times = f["VTKHDF/FieldData/Time"][()]
        time_indices = [0]
    elif fnames[0].endswith("pvd"):
        reader = pv.PVDReader(fnames[0])
        output_times = np.array(reader.time_values)
        time_indices = compute_time_indices(output_times, args.time.split(";"))
    else:
        raise NotImplementedError("only supported files are pvd, xdmf and hdf")
    filtered_list = []
    n_output_times = len(output_times)
    for x in time_indices:
        if -n_output_times <= x < n_output_times:
            filtered_list.append(x)
        else:
            print(f"Warning: time index {x} removed as out of range.")
    times = [output_times[k] for k in filtered_list]
    if not len(times):
        raise ValueError("all time index given are invalid")
    print(f"snapshots will be generated at times: {times}")

    def generate_snap(itime, mytime):
        plotter = pv.Plotter(off_screen=not args.interactive, **dic_window_size)
        for i, fname in enumerate(fnames):
            var = variables[i]
            if fname.endswith("xdmf"):
                sx = seissolxdmfExtended(fname)
                xyz = sx.ReadGeometry()
                connect = sx.ReadConnect()
                grid = create_vtk_grid(xyz, connect)
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
            elif fname.endswith("hdf"):
                reader = vtk.vtkHDFReader()
                reader.SetFileName(fname)
                reader.Update()
                grid = reader.GetOutputDataObject(0)
            elif fname.endswith("pvd"):
                reader = pv.PVDReader(fname)
                time_values = reader.time_values
                idx = compute_time_indices(output_times, [str(mytime)])
                if len(idx) == 0:
                    print(f"no output at t={mytime}s found for {fname}, skipping...")
                    continue
                reader.set_active_time_value(time_values[idx[0]])
                grid = reader.read()
                # compute slip-rate from slip-rate0 and slip-rate1
                for block in grid:
                    if (
                        isinstance(block, pv.DataSet)
                        and "slip-rate" == var
                        and "slip-rate0" in block.point_data
                        and "slip-rate1" in block.point_data
                    ):
                        slip_rate0 = block.point_data["slip-rate0"]
                        slip_rate1 = block.point_data["slip-rate1"]
                        slip_rate_magnitude = np.sqrt(slip_rate0**2 + slip_rate1**2)
                        block["slip-rate"] = slip_rate_magnitude
                    if var in block.point_data:
                        block[var] = block.point_data[var]
                    elif var in block.cell_data:
                        block[var] = block.cell_data[var]
            else:
                raise NotImplementedError("only supported files are xdmf, hdf, and pvd")

            mesh = pv.wrap(grid)

            if args.slice:
                args_slice = [float(v) for v in args.slice[0].split()]
                enabled_slice = [int(v) for v in args.slice[1].split(";")]
                if enabled_slice[i]:
                    assert len(args_slice) == 6
                    px, py, pz, nx, ny, nz = args_slice
                    mesh = mesh.slice(
                        normal=(nx, ny, nz),
                        origin=(px, py, pz),
                        generate_triangles=True,
                    )
                    assert mesh.n_points > 0

            clim_dic = color_ranges[i] if args.color_ranges else {"clim": None}

            if args.scalar_bar:
                sb_args = args.scalar_bar.split()
                height_pxl = int(sb_args[2]) if len(sb_args) == 3 else 150
                xr, yr = [float(v) for v in sb_args[0:2]]
                height = height_pxl / args.window_size[1]
                width = 40 / args.window_size[0]
                # shift successive scalar bars
                xr += 2 * width * i

                def get_scalar_bar_title(var):
                    if var == "SR":
                        return "slip rate (m/s)"
                    elif var == "ASl":
                        return "fault slip (m)"
                    elif var == "Vr":
                        return "rupture speed (m/s)"
                    elif var == "Vr_kms":
                        return "rupture speed (km/s)"
                    elif var == "PSR":
                        return "peak slip-rate (m/s)"
                    elif var == "mu_s":
                        return "static friction"
                    elif var == "d_c":
                        return "slip weakening distance (m)"
                    elif var in ["shear_stress_MPa", "shear_stress0_MPa"]:
                        return "shear stress (MPa)"
                    else:
                        return var

                scalar_bar_args = dict(
                    width=width,
                    height=height,
                    vertical=True,
                    position_x=xr,
                    position_y=yr,
                    label_font_size=int(1.8 * args.font_size),
                    title_font_size=int(1.8 * args.font_size),
                    n_labels=3,
                    fmt="%.1e" if use_log_scale[i] else "%g",
                    title=get_scalar_bar_title(var),
                )
                scalar_bar_dic = {"scalar_bar_args": scalar_bar_args}
            else:
                scalar_bar_dic = {}

            plotter.add_mesh(
                mesh,
                cmap=cmaps[i],
                scalars=var,
                **lighting,
                log_scale=use_log_scale[i],
                opacity=opacity[i],
                **clim_dic,
                **scalar_bar_dic,
            )

            if args.contours:
                add_contours(plotter, grid, sx, i, idx[0], args.contours)

            if not args.scalar_bar:
                plotter.remove_scalar_bar()
            is_surface = grid is vtk.vtkPolyData
            if (
                (not args.hide_boundary_edges)
                and ("surface" not in fname)
                and is_surface
            ):
                edges = mesh.extract_feature_edges(
                    boundary_edges=True,
                    feature_edges=False,
                    manifold_edges=False,
                    non_manifold_edges=False,
                )
                if edges.n_points > 0:
                    plotter.add_mesh(edges, color="k", line_width=2)

        if args.vtk_meshes:
            list_vtk_mesh_args = args.vtk_meshes.split(";")
            for vtk_mesh_args in list_vtk_mesh_args:
                fname, color, line_width = vtk_mesh_args.split()
                vtk_mesh = pv.read(fname)
                plotter.add_mesh(vtk_mesh, color=color, line_width=int(line_width))
        configure_camera(plotter, mesh, args.view)

        if args.zoom:
            plotter.camera.zoom(args.zoom)

        if args.annotate_time:
            colname, xr, yr = args.annotate_time.split()
            x1 = float(xr) * args.window_size[0]
            y1 = float(yr) * args.window_size[1]

            plotter.add_text(
                f"{mytime:.1f}s",
                position=(x1, y1),
                color=colname,
                font_size=args.font_size,
            )

        if args.annotate_text:
            annot_str = args.annotate_text.split(";")
            for params in annot_str:
                parts = params.split(" ", 3)
                assert (
                    len(parts) == 4
                ), f"Invalid format. Expected 'color x y text', got {parts}"
                colname, xr, yr, text_part = parts
                x1 = float(xr) * args.window_size[0]
                y1 = float(yr) * args.window_size[1]

                text_part = text_part.replace("\\n", "\n")
                # add time if {t} in the text
                if "{t" in text_part:
                    formatted_time = format_time(mytime)
                    text_part = text_part.format(t=formatted_time)

                plotter.add_text(
                    text_part,
                    position=(x1, y1),
                    color=colname,
                    font_size=args.font_size,
                )

        if args.interactive:
            plotter.show()
        else:
            out_fname = get_snapshot_fname(args, fname, itime, mytime)
            plotter.screenshot(out_fname)
            print(f"done writing {out_fname}")
        plotter.close()
        plotter.deep_clean()

    for itime, mytime in enumerate(times):
        generate_snap(itime, mytime)


if __name__ == "__main__":
    main()
