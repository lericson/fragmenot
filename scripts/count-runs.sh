#!/bin/bash

var="${var-var}"

shopt -s nullglob

while getopts "d:" opt; do
  case "$opt" in
    d) var="${OPTARG%/}";;
    ?) exit 1
  esac
done

for d in ./"${var}"/runs_*; do
  f=( "$d"/run_* )
  printf "%-20s %d %s\n" "$d:" "${#f[@]}"
done
