# Lightsheet Volume Analysis

## Requirements

Matlab (tested with R2017b, R2018a)

## Usage

- Main function: "VolumeMain"
- The results are written into an excel file in the specified output folder.
- For each time a 3D volume rendering figure is also saved in output folder. 

The calculated metrics are
- Total_Area:	Area of the segmentations Z-projection in um.
- Total_Volume:	Volume of segmentations in um3.
- majoraxis: returns the length of a unit vector that points in the direction of the major axis
