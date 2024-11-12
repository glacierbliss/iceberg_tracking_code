#!/bin/bash 

# -x

# This creates a time-lapse video that works in media player

workspace=$1
xscale=$2
yscale=$3
videoname=$4
framenumber=$5
searchpattern=$6

cd $workspace

mencoder mf://$searchpattern -mf fps=$framenumber -o $videoname -ovc lavc -lavcopts vcodec=msmpeg4v2:vbitrate=20000000 -vf scale=$xscale:$yscale

# mencoder mf://*png -mf fps=$framenumber -o $videoname -ovc lavc -lavcopts vcodec=msmpeg4v2:vbitrate=20000000 -vf scale=1000:-10