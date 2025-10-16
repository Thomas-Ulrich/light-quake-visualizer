# Light Quake Visualizer

A collection of scripts for visualizing output from earthquake simulation software.
Currently, it is designed to handle outputs from
[SeisSol](https://github.com/seissol/) and
[Tandem](https://github.com/TEAR-ERC/tandem).

## Features

- Visualize SeisSol and Tandem output files in XDMF, HDF-VTK, and PVD formats
- Support for plotting multiple datasets and variables simultaneously
- Customizable color maps, color ranges, and scalar bar settings
- Adjustable lighting and zoom for 3D visualization
- Flexible slicing options
- Contour plotting, e.g. for rupture time

For a full list of available options, run:

```bash
light_quake_visualizer --help
```

## Simple Example

The following command plots a volume output file at time 10s, variable `u`,
using a saved ParaView view (`tpv5.pvcc`):

```bash
light_quake_visualizer \
  output_tpv5_ref/tpv5_sym.xdmf \
  --var u \
  --time 10.0 \
  --cmap broc \
  --view output_tpv5_ref/tpv5.pvcc \
  --scalar_bar "0.9 0.1" \
  --color_range "-0.5 0.5" \
  --zoom 1.0 \
  --lighting 0.6 0.4 0.6 \
  --annotate_text "black 0.1 0.9 {time:.1f}"
```

## Plotting Multiple Datasets

Example showing a sliced volume output (`u`) and an unsliced fault output (`ASl`):

```bash
light_quake_visualizer \
  "output_tpv5_ref/tpv5_sym.xdmf;output_tpv5_ref/tpv5_sym-fault.xdmf" \
  --var "u;ASl" \
  --time 10.0 \
  --cmap "broc;viridis" \
  --view output_tpv5_ref/tpv5.pvcc \
  --scalar_bar "0.8 0.1" \
  --color_range "-0.5 0.5;0 5" \
  --zoom 1.0 \
  --light 0.5 0.5 0.5 \
  --slice "0 0 -2000 0 0 1" "1;0"
```

## Plotting Rupture Time Contours

Plot fault slip (`ASl`) with rupture time (`RT`) contour lines:

```bash
light_quake_visualizer \
  --variable ASl \
  --cmap davos_r0 \
  --color_range "0 3.0" \
  --contour \
    "file_index=0 var=RT contour=grey,2,0,max,1 contour=black,4,0,max,5" \
  --zoom 2.0 \
  --window 1200 600 \
  --output ASl \
  --time "i-1" \
  --view normal \
  output_tpv5_ref/tpv5_sym.xdmf
```

## Support for the New VTKHDF Format

Example usage with the new VTKHDF format:

```bash
light_quake_visualizer \
  output_tpv5_new_format/tpv5_sym-wavefield-2.hdf \
  --var u \
  --time "i0" \
  --cmap broc \
  --view output_tpv5_ref/tpv5.pvcc \
  --scalar_bar "0.9 0.1" \
  --color_range "-0.5 0.5" \
  --zoom 1.0 \
  --lighting 0.6 0.4 0.6
```

## Tandem Fault Output Example

Example usage for a 3D Tandem fault output:

```bash
light_quake_visualizer \
  --time "i::5" \
  --var slip-rate \
  --cmap turbo \
  output/fault.pvd \
  --view xz \
  --log_scale 0 \
  --scalar_bar "0.1 0.3" \
  --annotate_text "black 0.1 0.9 {long_time}" \
  --zoom 1.5 \
  --color_range "1e-7 1e0"
```

## Generate Vector Graphic Color Bars

You can generate standalone vector graphic color bars using `generate_color_bar`.
An example usage is provided below:

```bash
generate_color_bar vik --crange -2 2 --labelfont 8 --height 1.2 3.6 --nticks 3
```

## Combining Snapshots with Partial Overlap

You can combine multiple snapshots with adjustable overlap using `image_combiner`.
An example usage is provided below:

```bash
image_combiner \
  --i image1.png image2.png \
  --o combined_image.png \
  --col 2 \
  --rel 0.5 1.0
```
