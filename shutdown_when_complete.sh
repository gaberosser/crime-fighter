#!/bin/bash

if [ $EUID != 0 ]; then
    sudo "$0" "$@"
    exit $?
fi

if [ ! -f "$1" ]
    then
    echo "ERROR: File $1 does not exist" 1>&2
    exit 1
fi

while true; do
    if [ ! -f "$1" ]
    then
        echo "SHUTDOWN"
        sleep 10
        shutdown now -h
    fi
    echo "SLEEPING"
    sleep 600
done
