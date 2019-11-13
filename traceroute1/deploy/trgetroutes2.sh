#!/bin/sh
name=$(cat /etc/salt/minion_id)
datafile=./trdata2/trdata2.txt.$name

list=$(cat trdestinations.csv | grep $name | awk -F"," '{print $3}')

if [ -f "$datafile" ]; then
	rm -f $datafile
fi

for i in $list; do
	for j in $(seq 1 3);do
		echo "$j<<" >> $datafile
		traceroute -m 40 $i >> $datafile 2>&1
		echo "$j<<" >> $datafile
		ping -c 5 $i >> $datafile 2>&1
	done
done
