#PBS -V

### Set the job name
#PBS -N {{ JOB_NAME }}

### Join the output and error streams as standard output
#PBS -j oe

### Specify output directory
#PBS -o {{ STD_ERR_OUT }}

### Declares the shell that interprets the job script
#PBS -S /bin/sh

### Specify the number of cpus for your job. 1 processor on 1 node for benchmarking
### Also specifies to pick any node brncl-01 through brncl-32
#PBS -l nodes=1:ppn={{ PPN }}:intel

### Tell PBS how much memory you expct to use. Use units of 'b','kb', 'mb' or 'gb'.
#PBS -l mem={{ GB_MEM }}gb

### Tell PBS the anticipated run-time for your job, where walltime=HH:MM:SS
#PBS -l walltime={{ WALLTIME_HRS }}:00:00

### Switch to the working directory; by default TORQUE launches processes
### from your home directory.

cd $PBS_O_WORKDIR
source activate qiime2-2020.8
echo Working directory is $PBS_O_WORKDIR

# Calculate the number of processors allocated to this run.
NPROCS=`wc -l < $PBS_NODEFILE`

# Calculate the number of nodes allocated.
NNODES=`uniq $PBS_NODEFILE | wc -l`

### Display the job context
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Using ${NPROCS} processors across ${NNODES} nodes

input={{ PARAMS_FP }}
chunk_size={{ CHUNK_SIZE }}

# Calculate chunk sizes and the index of the last chunk
max_size="$(wc -l <${input})"
max_n_chunks=$((($max_size/$chunk_size)+1))
offset=$(($chunk_size * $PBS_ARRAYID))

# If we are on the last possible chunk, set its size
# to the number of remaining params
if [ ${PBS_ARRAYID} = ${max_n_chunks} ]
then
    final_chunk_size=$(($max_size%$chunk_size))
    previous_offset=$(($chunk_size * ($PBS_ARRAYID-1)))
    offset=$(($previous_offset+$final_chunk_size))
    chunk_size=$final_chunk_size
fi

echo Chunk number ${PBS_ARRAYID} size is ${chunk_size}

SUBSETLIST={{ RESULTS_DIR }}/subset${PBS_ARRAYID}.list

head -n $offset $input | tail -n $chunk_size > ${SUBSETLIST}

FORCE={{ FORCE_OVERWRITE }}
while IFS=$'\t' read -r idx params
do 
    RESULTS={{ RESULTS_DIR }}/${idx}_chunk_${PBS_ARRAYID}
    if [[ -f $RESULTS.qza && ${FORCE} = false ]]
    then
        echo $RESULTS already exists, execution skipped
    else
        qiime mlab unit-benchmark --i-table {{ TABLE_FP }} \
            --i-metadata {{ METADATA_FP }} \
            --p-algorithm {{ ALGORITHM }} \
            --p-params "${params}" \
            --p-n-repeats {{ N_REPEATS }} \
            --o-result-table ${RESULTS} \
            --verbose
    fi
done < ${SUBSETLIST}
