#!/bin/bash
module purge
ml GCC/13.3.0
ml OpenMPI/5.0.3
ml SuiteSparse/7.7.0-METIS-5.1.0
export WATCH_SECONDS=1
export NUM_THREADS=96
cd /scratch/user/u.ks124812/wes_benchmarking/sparse_matrix_benchmark
make -B vector
for(( j=0 ; j<3 ; j++ ))
do
  for i in 1050000; do
    > ./meminfo_output_memverge_run_$j/meminfo_$i.log 
    > ./ps_output_memverge_run_$j/ps_output_memverge_$i.log
    > ./free_output_memverge_run_$j/free_output_memverge_$i.log
    watch -n $WATCH_SECONDS "cat /proc/meminfo >> ./meminfo_output_memverge_run_$j/meminfo_$i.log; date +'%Y-%m-%dT%H:%M:%S%z' >> ./meminfo_output_memverge_run_$j/meminfo_$i.log" &> /dev/null &
    watch -n $WATCH_SECONDS "ps -p $(pgrep -f "matrix_vector") -o %cpu,%mem,cmd --no-headers >>./ps_output_memverge_run_$j/ps_output_memverge_$i.log; date +'%Y-%m-%dT%H:%M:%S%z' >> ./ps_output_memverge_run_$j/ps_output_memverge_$i.log" &> /dev/null &
    watch -n $WATCH_SECONDS "free -h >> ./free_output_memverge_run_$j/free_output_memverge_$i.log; date +'%Y-%m-%dT%H:%M:%S%z' >> ./free_output_memverge_run_$j/free_output_memverge_$i.log" &> /dev/null &
        echo n: $i
        echo Start time:
        date
        ./matrix_vector $i $NUM_THREADS
        echo End time:
        date
        killall -9 watch
        sleep 5
  done
done
killall -9 watch

