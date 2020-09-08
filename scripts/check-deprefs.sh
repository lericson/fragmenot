#!/bin/bash

set -e

for GIT_WORK_TREE in ./env/src/*; do
  GIT_DIR=$GIT_WORK_TREE/.git
  DEPNAME=$(basename $GIT_WORK_TREE)
  DEPREF=refs/heads/deps/$DEPNAME
  if ! git rev-parse $DEPREF >/dev/null 2>&1; then
    continue
  fi
  if ! GIT_WORK_TREE=$GIT_WORK_TREE GIT_DIR=$GIT_DIR git diff --quiet; then
    echo $GIT_WORK_TREE has uncommitted changes
  fi;
  if [[ $(git rev-parse $DEPREF) != $(git --git-dir=$GIT_DIR rev-parse HEAD) ]]; then
    echo $GIT_WORK_TREE HEAD is not equal to $DEPREF
    git fetch -n $GIT_DIR +HEAD:$DEPREF
  fi;
done

REQUIREMENTS=requirements.txt
awk -F '[ /@#=]' '$1 == "-e" && $2 == "git+file:" { sub("_", "-", $7); print NR, $5, $7 }' <$REQUIREMENTS | \
while read -ra LINE; do
  LNO=${LINE[0]}
  COMMIT=${LINE[1]}
  DEPNAME=${LINE[2]}
  DEPREF=refs/heads/deps/$DEPNAME
  GIT_WORK_TREE=./env/src/$DEPNAME
  GIT_DIR=$GIT_WORK_TREE/.git
  if [ ! -e $GIT_WORK_TREE ]; then
    echo $GIT_WORK_TREE does not exist, you need to re-run pip install -r requirements.txt
  fi
  if ! git rev-parse $DEPREF >/dev/null 2>&1; then
    echo $DEPREF does not exist, you need to fetch deps references
    continue
  fi
  if [[ $(git rev-parse $DEPREF) != $(git --git-dir=$GIT_DIR rev-parse $COMMIT) ]]; then
    echo $REQUIREMENTS:$LNO: Commit $COMMIT is not equal to $DEPREF '('$(git rev-parse $DEPREF)')'
  fi
done

if ! git diff --quiet; then
  echo Main repository has uncommited changes
fi
