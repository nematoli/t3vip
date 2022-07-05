#!/bin/bash

# Download, Unzip, and Remove zip
if [ "$1" = "dexhand" ]
then

    echo "Downloading dexhand ..."
    wget http://t3vip.cs.uni-freiburg.de/dataset/dexhand.zip
    unzip dexhand.zip && rm dexhand.zip
    echo "saved folder: dexhand"
elif [ "$1" = "Omnipush" ]
then

    echo "Downloading Omnipush ..."
    wget http://t3vip.cs.uni-freiburg.de/dataset/Omnipush.zip
    unzip Omnipush.zip && rm Omnipush.zip
    echo "saved folder: Omnipush"

else
    echo "Failed: Usage download_data.sh dexhand | Omnipush | Calvin"
    exit 1
fi