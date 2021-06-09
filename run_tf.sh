#!/bin/bash

export JAVA_HOME=/usr/java/default
export HADOOP_HDFS_HOME=/usr/lib/hadoop-hdfs
export HADOOP_COMMON_HOME=/usr/lib/hadoop
export HADOOP_MAPRED_HOME=/usr/lib/hadoop-mapreduce
export HADOOP_CONF_DIR=/etc/hadoop/conf
export LD_LIBRARY_PATH=/usr/lib/hadoop/lib/native:/usr/java/default/jre/lib/amd64/server
export CLASSPATH=`hadoop classpath --glob`

python tf_cluster.py