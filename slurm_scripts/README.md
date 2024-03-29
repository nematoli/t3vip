##	Training T3VIP on a Slurm Cluster
### Starting a training
```bash
$ cd $T3VIP_ROOT/slurm_scripts
$ python slurm_training.py --venv t3vip_venv --train_file "../t3vip/train.py"
```
This assumes that `--venv t3vip_venv` specifies a conda environment.
To use virtualenv instead, change line 18 of sbatch_train.sh and sbatch_eval.sh accordingly.

All hydra arguments can be used as in the normal training.

Use the following optional command line arguments for slurm:
- `--log_dir`: slurm log directory
- `--job_name`: slurm job name
- `--gpus`: number of gpus
- `--mem`: memory
- `--cpus`: number of cpus
- `--days`: time limit in days
- `--partition`: name of slurm partition

The script will create a new folder in the specified log dir with a date tag and the job name.
This is done *before* the job is submitted to the slurm queue.
In order to ensure reproducibility, the current state of the t3vip repository
is copied to the log directory at *submit time* and is
locally installed, such that you can schedule multiple trainings and there is no interference with
future changes to the repository.

### Resuming a training
Every job submission creates a `resume_training.sh` script in the log folder. To resume a training,
call `$ sh <PATH_TO_LOG_DIR>/resume_training.sh`. By default, the model loads the latest saved checkpoint.

### Evaluating a model
To evaluate a trained model via slurm, run `$ sh <PATH_TO_LOG_DIR>/evaluate.sh`, which will automatically place a job 
on the same partition as it was trained on. Note that this script is also autogenerated.
