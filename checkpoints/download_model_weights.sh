#!/bin/bash
# Download, Unzip, and Remove zip
if [ "$1" = "dexhand" ]
then

    echo "Downloading T3VIP Checkpoint for dexhand dataset ..."
    wget http://t3vip.cs.uni-freiburg.de/model_weights/dexhand.zip
    unzip dexhand.zip && rm dexhand.zip
    echo "finished!"
elif [ "$1" = "omni_s0" ]
then

    echo "Downloading T3VIP Checkpoint for omnipush dataset without skipping frames ..."
    wget http://t3vip.cs.uni-freiburg.de/model_weights/omnipush_s0.zip
    unzip omnipush_s0.zip && rm omnipush_s0.zip
    echo "finished!"

elif [ "$1" = "omni_s2" ]
then

    echo "Downloading T3VIP Checkpoint for omnipush dataset skipping 2 frames ..."
    wget http://hulc.cs.uni-freiburg.de/model_weights/omnipush_s2.zip
    unzip omnipush_s2.zip && rm omnipush_s2.zip
    echo "finished!"

elif [ "$1" = "calvin_s0" ]
then

    echo "Downloading T3VIP Checkpoint for calvin dataset without skipping frames ..."
    wget http://hulc.cs.uni-freiburg.de/model_weights/calvin_s0.zip
    unzip calvin_s0.zip && rm calvin_s0.zip
    echo "finished!"

elif [ "$1" = "calvin_s2" ]
then

    echo "Downloading T3VIP Checkpoint for calvin dataset skipping 2 frames ..."
    wget http://hulc.cs.uni-freiburg.de/model_weights/calvin_s2.zip
    unzip calvin_s2.zip && rm calvin_s2.zip
    echo "finished!"

else
    echo "Failed: Usage download_model_weights.sh dexhand | omnipush_s0 | omnipush_s2 | calvin_s0 | calvin_s2"
    exit 1
fi