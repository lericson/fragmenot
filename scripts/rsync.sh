#!/bin/sh
host=science2.lericson.se
module=covis19
source="rsync://${host}/${module}/var"
target=rsync
exec rsync -rt --info=progress2 "$source" "$target"
