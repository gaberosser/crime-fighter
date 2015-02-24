#!/bin/bash

if [ $EUID != 0 ]; then
    sudo "$0" "$@"
    exit $?
fi

if [ -f "/home/gabriel/signal/shut_me_down_goddamnit" ]
    then
    rm /home/gabriel/signal/shut_me_down_goddamnit
fi

while true; do
    if [ -f "/home/gabriel/signal/shut_me_down_goddamnit" ]
    then
        echo "SHUTDOWN"
        sleep 10
        shutdown now -h
    fi
    echo "SLEEPING"
    sleep 600
done