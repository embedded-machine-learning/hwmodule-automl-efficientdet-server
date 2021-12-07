#!/bin/bash

# put in home

echo "WARNING: Start script with . /[PATH_TO_SCRIPT], not with ./"

echo "Activate TF2 Object Detection API Python Environment"

PROJECTROOT=`pwd`
ENVROOT=../..

source $ENVROOT/tf24_py36/bin/activate
cd $ENVROOT/models/research/
#export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
export PYTHONPATH=`pwd`:`pwd`/slim
echo New python path $PYTHONPATH

cd $PROJECTROOT

echo "Activation complete"
