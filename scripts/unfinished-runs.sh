#!/bin/bash

var="${var-var}"

while getopts "d:" opt; do
  case "$opt" in
    d) var="${OPTARG%/}";;
    ?) exit 1
  esac
done

for d in ./"${var}"/runs_*/run_*; do
  f=( "$d"/state?????.npz )
  [[ "${#f[@]}" -lt 10 ]] && echo "$d"
done
