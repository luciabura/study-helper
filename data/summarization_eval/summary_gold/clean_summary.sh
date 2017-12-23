#!/usr/bin/env bash

N=1; for X in *summary.txt; do mv $X ${N}_summary.txt; N=$(($N+1)); done

