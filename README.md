# About
This repository provides Python code to track icebergs on oblique time lapse photos, to project the derived velocities from camera coordinates to map coordinates, and to postprocess the velocities in map coordinates (e.g., for spatial and temporal averaging). The tracking is based on the sparse Lucas-Kanade tracking algorithm implemented in OpenCV.
The code was developed as part of a National Science Foundation (NSF) funded research project at LeConte Glacier, Alaska. See the paper "Tracking icebergs with time-lapse photography and sparse optical flow, LeConte Bay, Alaska, 2016–2017", published in the Journal of Glaciology, for details.
[credit to Christian here]
# Dependencies
[details for container creation with docker here] [details for dependencies in a requirements file here]
# Usage
The code is written in Python 2.7 and tested in an Ubuntu 16.04 environment. OpenCV version 3.1.0. is used. There are several other dependencies - check script imports. The scripts are named hierarchically, representing individual steps of the workflow.
The tracking algorithm requires time-lapse sequences with high temporal resolution to work best - ideally one or more photos per minute. Run the "0_1_test_lucaskanade_tracking.py" script to evaluate applicability to your time-lapse sequence.
