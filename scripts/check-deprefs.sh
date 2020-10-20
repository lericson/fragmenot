#!/bin/bash

set -e

for GIT_WORK_TREE in ./env/src/*; do
  GIT_DIR=$GIT_WORK_TREE/.git
  DEPNAME=$(basename $GIT_WORK_TREE)
  DEPREF=refs/heads/deps/$DEPNAME
  printf "Checking repository $GIT_WORK_TREE... "
  if ! git rev-parse $DEPREF >/dev/null 2>&1; then
    printf "Not a depref"
    continue
  fi
  if ! GIT_WORK_TREE=$GIT_WORK_TREE GIT_DIR=$GIT_DIR git diff --quiet; then
    printf "Uncommitted changes!\n"
    continue
  else
    printf "No uncommited changes... "
  fi
  if [[ $(git rev-parse $DEPREF) != $(git --git-dir=$GIT_DIR rev-parse HEAD) ]]; then
    printf "HEAD is not equal to $DEPREF!\n"
    printf "I can fix this for you.\n"
    printf "Press Enter to set $DEPREF to the HEAD of $GIT_WORK_TREE\n"
    read -r || exit
    git fetch -n $GIT_DIR +HEAD:$DEPREF
  else
    printf "Seems OK.\n"
  fi
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
  printf "Checking requirement for $DEPREF... "
  if [ ! -e $GIT_WORK_TREE ]; then
    printf "\n%s does not exist, you need to re-run pip install -r requirements.txt" "$GIT_WORK_TREE"
  elif ! git rev-parse $DEPREF >/dev/null 2>&1; then
    printf "\n%s does not exist, you need to fetch deps references\n" "$DEPREF"
  elif [[ $(git rev-parse $DEPREF) != $(git --git-dir=$GIT_DIR rev-parse $COMMIT) ]]; then
    printf "\n%s: Commit %s is not equal to %s (%s)\n" "$REQUIREMENTS:$LNO" "$COMMIT" "$DEPREF" "$(git rev-parse $DEPREF)"
  else
    printf "Seems OK.\n"
  fi
done

if ! git diff --quiet; then
  printf "Main repository has uncommited changes!\n"
fi
