#!/bin/bash

echo WARNING: Start script with . /[PATH_TO_SCRIPT], not with ./


END=6
for ((i=0;i<=END;i++)); 
do 
  echo Download Weight size $i from https://github.com/google/automl/tree/master/efficientdet
  wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d$i.tar.gz
  tar xf efficientdet-d$i.tar.gz
  rm efficientdet-d$i.tar.gz
done

END=4
for ((i=0;i<=END;i++)); 
do 
  echo Download Weight size $i from lite models
  wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-lite$i.tgz
  tar xf efficientdet-lite$i.tgz
  rm efficientdet-lite$i.tgz
done

wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-lite3x.tgz
tar xf efficientdet-lite3x.tgz
rm efficientdet-lite3x.tgz

