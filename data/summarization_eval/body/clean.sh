#!/usr/bin/env bash

N=1; for X in *body*txt; do mv $X ${N}_body.txt; N=$(($N+1)); done

