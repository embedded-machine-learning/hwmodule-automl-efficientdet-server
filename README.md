# Training and Inference of AutoML EfficientDet on a Linux Server with EML Tools
In this folder, there is a template project for inference of trained, exported models of AutoML EfficientDet on a training server. In the following
procedure, instructions are provided to setup and run one or more networks and to extract the evaluations of the executions. All
evaluations are compatible with the EML tools.

## Setup

### Prerequisites
1. Setup the task spooler on the target device. Instructions can be found here: https://github.com/embedded-machine-learning/scripts-and-guides/blob/main/guides/task_spooler_manual.md

### Dataset
For validating the tool chain, download the small validation set from kaggle: https://www.kaggle.com/alexanderwendt/oxford-pets-cleaned-for-eml-tools

It contains of two small sets that are used for training and inference validation in the structure that is compatible to the EML Tools. 
Put it in the following folder structure, e.g. ```/srv/cdl-eml/datasets/dataset-oxford-pets-cleaned/```

### Generate EML Tools directory structure and Setup the TF2ODA Environment
The following steps are only necessary if you setup the EML tools for the first time on a device.
1. Create a folder for your datasets. Usually, multiple users use one folder for all datasets to be able to share them. Later on, in the 
training and inference scripts, you will need the path to the dataset.

2. Create the EML tools folder structure, e.g. ```eml-tools```. The structure can be found here: https://github.com/embedded-machine-learning/eml-tools#interface-folder-structure. 
Most of the following steps are performed with this script as well: ```generate_workspace_efficientdet_server.sh```. The script generates two environments:
First, an AutoML environment and second, a TF2ODA environenment from which some methods are used for training and evaluation. You will need both environments
because the ```utils``` package exist in both AutoML and TF2ODA and the versions do not cope well with each other.

```
#!/bin/bash


#1. Create a folder for your datasets. Usually, multiple users use one folder for all datasets to be able to share them. Later on, in the 
#training and inference scripts, you will need the path to the dataset.
#2. Create the EML tools folder structure, e.g. ```eml-tools```. The structure can be found here: https://github.com/embedded-machine-learning/eml-tools#interface-folder-structure
ROOTFOLDER=`pwd`

#In your root directory, create the structure. Sample code
mkdir -p eml_projects
mkdir -p venv

#3. Clone the EML tools repository into your workspace
EMLTOOLSFOLDER=./eml-tools
if [ ! -d "$EMLTOOLSFOLDER" ] ; then
  git clone https://github.com/embedded-machine-learning/eml-tools.git "$EMLTOOLSFOLDER"
else 
  echo $EMLTOOLSFOLDER already exists
fi

# Project setup
#4. Clone the AutoML repository
AUTOMLFOLDER=./automl
if [ ! -d "$AUTOMLFOLDER" ] ; then
  git clone https://github.com/google/automl.git "$AUTOMLFOLDER"
else 
  echo $AUTOMLFOLDER already exists
fi

#5. Create a virtual environment for TF2ODA in your venv folder. The venv folder is put outside of the project folder to 
#avoid copying lots of small files when you copy the project folder. Conda would also be a good alternative.
# From root
cd $ROOTFOLDER

cd ./venv

EFFDETENV=effdet_py36
if [ ! -d "$EFFDETENV" ] ; then
  virtualenv -p python3.8 $EFFDETENV
  source ./$EFFDETENV/bin/activate

  # Install necessary libraries
  python -m pip install --upgrade pip
  pip install --upgrade setuptools cython wheel
  
  # Install EML libraries
  pip install lxml xmltodict tdqm beautifulsoup4 pycocotools numpy tdqm pandas matplotlib pillow
  
  # Install TF2ODA specifics
  #pip install tensorflow==2.4.1
  
  cd $ROOTFOLDER
  
  pip install -r ./$AUTOMLFOLDER/efficientdet/requirements.txt
  
  cd $ROOTFOLDER/automl/efficientdet
  export PYTHONPATH=$PYTHONPATH:`pwd`
  echo New python path $PYTHONPATH
  
  cd $ROOTFOLDER

  echo # Test installation
  # If all tests are OK or skipped, then the installation was successful
  #python object_detection/builders/model_builder_tf2_test.py
  
  echo Test if Tensorflow works with CUDA on the machine. For TF2.4.1, you have to use CUDA 11.0
  python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
  
  echo Installation complete
  
else 
  echo $EFFDETENV already exists
fi

cd $ROOTFOLDER
source ./venv/$EFFDETENV/bin/activate


# Create TF2ODA environment that is used for the evaluation and inference measurements
TF2ODAENV=tf24_py36
if [ ! -d "$TF2ODAENV" ] ; then
  virtualenv -p python3.8 $TF2ODAENV
  source ./$TF2ODAENV/bin/activate

  # Install necessary libraries
  python -m pip install --upgrade pip
  pip install --upgrade setuptools cython wheel
  
  # Install EML libraries
  pip install lxml xmltodict tdqm beautifulsoup4 pycocotools numpy tdqm pandas matplotlib pillow
  
  cd $ROOTFOLDER
  
  echo # Install protobuf
  PROTOC_ZIP=protoc-3.14.0-linux-x86_64.zip
  curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v3.14.0/$PROTOC_ZIP
  unzip -o $PROTOC_ZIP -d protobuf
  rm -f $PROTOC_ZIP
  
  echo # Clone tensorflow repository
  git clone https://github.com/tensorflow/models.git
  cd models/research/
  cp object_detection/packages/tf2/setup.py .
  python -m pip install .
  
  echo # Add object detection and slim to python path
  export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
  
  echo # Prepare TF2 Proto Files
  ../../protobuf/bin/protoc object_detection/protos/*.proto --python_out=.

  echo # Test installation
  # If all tests are OK or skipped, then the installation was successful
  python object_detection/builders/model_builder_tf2_test.py
  
  # Install TF2ODA specifics
  #pip install tensorflow==2.4.1
  echo #Test if Tensorflow works with CUDA on the machine. For TF2.4.1, you have to use CUDA 11.0
  python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
  
  echo "Important information: If there are any library errors, you have to install the correct versions manually. TFODAPI does install the latest version of "
  echo "tensorflow. However, in this script Tensorflow 2.4.1 is desired. Then, you have to uninstall the newer versions and replace with current versions."

  echo # Installation complete
  
else 
  echo $TF2ODAENV already exists
fi

cd $ROOTFOLDER
source ./venv/$EFFDETENV/bin/activate

echo Created environment for AutoML EfficientDet Training and inference
```

### Project setup
1. Go to your project folder e.g. ```./eml_projects``` and create a project folder, e.g. ```./effdet-oxford-pets```

2. Copy the scripts from this repository to that folder and execute ```chmod 777 *.sh``` to be able to run the scripts. One of the script is the 
task spooler script, which could be used by multiple EML projects, ```./init_ts.sh```.

3. run ```./setup_dirs_effdet_server.sh``` to generate all necessary folders.

4. Download pretrained models: Go to ```./pre-trained-models```, copy and execute ```./download_checkpoints.sh``` from this repository ```./pre-trained-models```.
The script downloads a number of pretrained EfficientDet models from https://github.com/google/automl/tree/master/efficientdet

5. In ```./config```, create or copy the *.config files, e.g. ```tf2effdet_efficientdetd1_512x512_pets.yaml```, which you create 
from the template configs from the pretrained models. The creation of files can be found here: https://github.com/google/automl/tree/master/efficientdet#8-finetune-on-pascal-voc-2012-with-detector-coco-ckpt.
The names of the config files are the model names.


### Modification of script files
The next step is to adapt the script files to the current environment.

#### Adapt Task Spooler Script
In ```init_ts.sh```, either adapt

```
export TS_SOCKET="/srv/ts_socket/GPU.socket"
chmod 777 /srv/ts_socket/GPU.socket
export TS_TMPDIR=~/logs
``` 

to your task spooler path or call another task spooler script in your EML Tools root.
```
. ../../init_ts.sh
```

#### Adapt Environment Scripts
In ```init_env.sh```, adapt the following part to your venv folder or conda implementation.

```
PROJECTROOT=`pwd`
ENVROOT=../..

source $ENVROOT/venv/effdet_py36/bin/activate
cd $ENVROOT/automl/efficientdet
export PYTHONPATH=$PYTHONPATH:`pwd`
```

In ```init_env_tf2oda.sh```, adapt the following part to your venv folder or conda implementation to use the TF2ODA.

```
source $ENVROOT/tf24_py36/bin/activate
cd $ENVROOT/models/research/
#export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
export PYTHONPATH=`pwd`:`pwd`/slim
```


#### Training Script
The script ```tf2_effdet_train_export_inf_TEMPLATE.sh``` trains and exports the model in the \"TEMPLATE\" part of the file name. 

For training, TEMPLATE has to be replaced by the network name that shall be trained. In case you don not use the ```add_folder...``` scripts, you can manually prepare the scripts.
First copy ```tf2_effdet_train_export_inf_TEMPLATE.sh``` and rename it to fit 
your network, e.g. ```tf2_effdet_train_export_inf_tf2effdet_efficientdet-d0_512x512_pets.sh```. The network will use the model name to load
the config from ```./config```.

For each network to be trained, the following constants have to be adapted:
```
# Constant Definition
USERNAME=wendt  #Set your user name
USEREMAIL=alexander.wendt@tuwien.ac.at   #Set your email
SCRIPTPREFIX=../../eml-tools   #No need to change
#Training set, full dataset
DATASET_TRAINING=/srv/cdl-eml/datasets/dataset-oxford-pets-cleaned   #Set the training dataset
#Validation set for the training, full dataset
DATASET_VALIDATION=/srv/cdl-eml/datasets/dataset-oxford-pets-cleaned  #Set the valiadation dataset for training
#Validation set for the validation on end devices
DATASET_INFERENCE=/srv/cdl-eml/datasets/dataset-oxford-pets-val-debug   #Set the valiadation data (sub) set for the end devices
HARDWARENAME=TeslaV100    #Set hardware name
# Model type: The name is a value used to find the model to used for training. Default is efficientdet-d1. 
NUMBEREPOCHS=1     #Set number of epochs. Default=300
NUMBEREXAMPLES=2000    #Set number of examples per epoch
# Set this variable true if the network shall be trained, else only inference shall be performed
TRAINNETWORK=true    #Set true

```

Now, the script will train a network and perform inference of it


#### Add Folder Jobs
```add_folder_train_inference_job.sh``` loads all config names from ```./config/*.yaml```, which are the model names. Then it makes a copy of
```tf2_effdet_train_export_inf_TEMPLATE.sh``` and replaces ```TEMPLATE``` with the 
model name. Then, it adds these two scripts to the task spooler.

No script adaptions are necessary.

## Running the system
Run ```./add_folder_train_inference_job.sh``` to add all models to the task spooler. The result are trained, exported and inferred models that can be copied
to the embedded target devices. Additionally, inference is performed on that device.

## Common Problems

### Task Spooler Blocked
If the task spooler freezes or is blocked, the following error message is shown:

```
=== Init task spooler ===
Setup task spooler socket for GPU.
chmod: changing permissions of '/srv/ts_socket/GPU.socket': Operation not permitted
task spooler output directory: /home/wendt/logs
Task spooler initialized /srv/ts_socket/GPU.socket
(tf24) [wendt@eda02 graz-pedestrian]$ ts -l
c: cannot connect to the server
(tf24) [wendt@eda02 graz-pedestrian]$
```

The cause is the a user blocks the task spooler and nobody else has access rights. It has to be released by the user or a sudo-user.
The solution is to put the following command line into the task spooler script: ```chmod 777 /srv/ts_socket/GPU.socket```

### Windows Instead of Linux EOL in Files
Note: If you get this error: ``` /bin/bash^M: bad interpreter``` or other strange execution problems, then you might use Windows EOL. To correct it, change EOL to Unix.


## Embedded Machine Learning Laboratory

This repository is part of the Embedded Machine Learning Laboratory at the TU Wien. For more useful guides and various scripts for many different platforms visit 
our **EML-Tools**: **https://github.com/embedded-machine-learning/eml-tools**.

Our newest projects can be viewed on our **webpage**: **https://eml.ict.tuwien.ac.at/**
