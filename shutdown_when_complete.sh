#!/bin/bash

while true; do
    if [ -f "/home/gabriel/signal/shut_me_down_goddamnit" ]
    then
        echo "SHUTDOWN"
        shutdown now -h
    fi
    echo "SLEEPING"
    sleep 600
done