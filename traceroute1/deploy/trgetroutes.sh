#!/bin/sh
name=$(cat /etc/salt/minion_id)
datafile=./trdata/trdata.txt.$name
if [ -f "$datafile" ]; then
	rm -f $datafile
fi
for i in $(cat trsites.list); do
	for j in $(seq 1 5);do
		echo "$j<<" >> $datafile
		traceroute -m 120 $i >> $datafile 2>&1
		echo "$j<<" >> $datafile
		ping -c 5 $i >> $datafile 2>&1
	done
done
