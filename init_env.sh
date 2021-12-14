#!/bin/bash

echo WARNING: Start script with . /[PATH_TO_SCRIPT], not with ./

#echo "Setup Task Spooler"
#. /srv/cdl-eml/tf2odapi/init_eda_ts.sh

echo Activate TF2 AutoML EfficientDet Environment

PROJECTROOT=`pwd`
ENVROOT=../..

source $ENVROOT/venv/effdet_py36/bin/activate
cd $ENVROOT/automl/efficientdet
export PYTHONPATH=`pwd`
echo New python path $PYTHONPATH

cd $PROJECTROOT

echo Activation complete
