#!/usr/bin/env bash

for i in *.txt; do
	sed -i '/^\[/ d' $i
done
