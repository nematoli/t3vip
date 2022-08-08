#!/bin/bash

# Download, Unzip, and Remove zip
if [ "$1" = "dexhand" ]
then

    echo "Downloading dexhand ..."
    wget http://t3vip.cs.uni-freiburg.de/dataset/dexhand.zip
    unzip dexhand.zip && rm dexhand.zip
    echo "saved folder: dexhand"
elif [ "$1" = "omnipush" ]
then

    echo "Downloading omnipush ..."
    wget http://t3vip.cs.uni-freiburg.de/dataset/omnipush.zip
    unzip omnipush.zip && rm omnipush.zip
    echo "saved folder: omnipush"

elif [ "$1" = "calvin" ]
then
    mkdir calvin && cd "$_"
    echo "Downloading calvin Env C and D episode indexing ..."
    wget http://t3vip.cs.uni-freiburg.de/dataset/calvin/task_idx.zip
    unzip task_idx.zip && rm task_idx.zip
    echo "saved folder: task_idx"

    echo "Downloading calvin task_D_D ..."
    wget http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip
    unzip task_D_D.zip && rm task_D_D.zip
    mv task_A_A task_D_D
    echo "saved folder: task_D_D"

    echo "Downloading task_ABC_D ..."
    wget http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip
    unzip task_ABC_D.zip && rm task_ABC_D.zip
    mv task_BCD_A task_ABC_D
    echo "saved folder: task_ABC_D"
else
    echo "Failed: Usage download_data.sh dexhand | omnipush | calvin"
    exit 1
fi