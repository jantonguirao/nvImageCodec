#!/bin/bash

set -e

git config -f .gitmodules --get-regexp '^submodule\..*\.path$' |
    while read path_key local_path
    do
        url_key=$(echo $path_key | sed 's/\.path/.url/')
        branch_key=$(echo $path_key | sed 's/\.path/.branch/')
        url=$(git config -f .gitmodules --get "$url_key")
        branch=$(git config -f .gitmodules --get "$branch_key" || echo "")
        rm -rf $local_path;
        if [[ -z "$branch" ]]; then
            git submodule add $url $local_path 
        else
            git submodule add -b $branch $url $local_path
        fi 
        echo "$local_path has been initialized"
    done

