#!/bin/bash

# You can probably adapt this to whatever the latest version is.

if [[ "$(uname -mo)" != "x86_64 GNU/Linux" ]]; then
  echo "This installer is for GNU/Linux x84_64."
  echo "Go find your OS on"
  echo
  echo "  https://www.embree.org/downloads.html"
  echo
  exit 1
fi

url="${url:-https://github.com/embree/embree/releases/download/v3.11.0/embree-3.11.0.x86_64.linux.tar.gz}"
filename="${url##*/}"
var="${var:-var}"

if [[ ! -f "$var/$filename" ]]; then
  [[ -e "$var/$filename" ]] && exit 1
  mkdir -p "$var"
  echo "Downloading $url"
  curl -Lo "$var/$filename" "$url" || exit
else
  echo "Found $filename"
fi

install_dir="${install_dir:-/usr/local}"
exec tar --strip-components=1 --wildcards -xvzf "$var/$filename" -C "$install_dir" '*'/{bin,cmake,include,lib,share}/'*'
