# Light Quake Visualizer

A collection of scripts to visualize SeisSol output using PyVista.

## Features

- Visualize SeisSol output files in XDMF and HDF-VTK formats
- Support for multiple datasets and variables
- Customizable color maps, color ranges, and scalar bar settings, lighting
- Slicing options
- And more...

For more information on available options and their descriptions, run:

```
light_quake_visualizer --help
```

## A simple example of use

Plot the volume output file at time 10s, variable u with a pvcc (saved from ParaView):

```
light_quake_visualizer output_tpv5_ref/tpv5_sym.xdmf --var u --time 10.0 --cmap broc --view output_tpv5_ref/tpv5.pvcc  --scalar_bar "0.9 0.1" --color_range "-0.5 0.5" --zoom 1.0 --lighting 0.6 0.4 0.6
```

## Plotting several datasets

Here sliced volume output (variable u), and (unsliced) fault output (variable ASl):

```
light_quake_visualizer  "output_tpv5_ref/tpv5_sym.xdmf;output_tpv5_ref/tpv5_sym-fault.xdmf" --var "u;ASl" --time 10.0 --cmap "broc;viridis" --view output_tpv5_ref/tpv5.pvcc  --scalar_bar "0.8 0.1" --color_range "-0.5 0.5;0 5" --zoom 1.0 --light 0.5 0.5 0.5 --slice "0 0 -2000 0 0 1" "1;0"
```

## Plotting rupture time contour lines 

Here is an example of plotting fault slip (ASl) with rupture time (RT) contour lines
```
light_quake_visualizer --variable ASl --cmap davos_r0 --color_range "0 3.0" --contour "file_index=0 var=RT contour=grey,2,0,max,1 contour=black,4,0,max,5" --zoom 2.0 --window 1200 600 --output ASl --time "i-1" --view normal output_tpv5_ref/tpv5_sym.xdmf
```

## Support for the new HDF-VTK

Example usage with the new HDF-VTK format:
```
light_quake_visualizer  output_tpv5_new_format/tpv5_sym-wavefield-2.hdf --var u --time "i0" --cmap broc --view output_tpv5_ref/tpv5.pvcc  --scalar_bar "0.9 0.1" --color_range "-0.5 0.5" --zoom 1.0 --lighting 0.6 0.4 0.6
```

## Support for tandem fault output

```
light_quake_visualizer --time "i::5" --var slip-rate --cmap turbo output/fault.pvd --view xz --log_scale 0 --scalar_bar "0.1 0.3" --annotate_text "black 0.1 0.9 {t}" --zoom 1.5 --color_range "1e-7 1e0"
```

## Generate vector graphic color bar image

Example usage:

```
generate_color_bar vik --crange -2 2 --labelfont 8 --height 1.2 3.6 --nticks 3
```

## Combining snapshots with possible overlap

Change the background to white, and combine several images with partial overlap:

```
image_combiner --i image1.png image2.png --o combined_image.png --col 2 --rel 0.5 1.0
```
