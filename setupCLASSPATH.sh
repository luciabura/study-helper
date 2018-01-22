#!/usr/bin/env bash
for file in `find ~/part-II/stanford-corenlp-full-2017-06-09/ -name "*.jar"`; do export
CLASSPATH="$CLASSPATH:`realpath $file`"; done