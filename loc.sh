#!/bin/bash

git shortlog -s -- $1 | cut -c8- | while read i
do
    git log --author="$i" --pretty=tformat: --numstat -- $1 \
    | awk -v name="$i" '{ add += $1; subs += $2; loc += $1 - $2 } END { printf "%s: added lines: %s removed lines: %s total lines: %s\n", name, add, subs, loc }'
done