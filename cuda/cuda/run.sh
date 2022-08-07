#!/bin/bash

#mpirun -np $1 --allow-run-as-root --bind-to none --hostfile $2 tools/dist_simulator
mpirun -np $1  --bind-to socket --hostfile $2 tools/dist_simulator -timeout=2
