# 508 Birds

Helpful commands:
* Show my jobs: `crc-squeue.py` or `crc-squeue.py -w` to watch continuously
* Cancel a job (find its PID first using one of the above commands): 
* Show someone else's jobs: `squeue -M gpu -u bmooreii`
* Get a list of partitions: `sinfo -M gpu`. Anything that doesn't say 'mix' could be better. (Note: can only use batch size of 64 on the `v100`.)
* See here under "CRC wrappers" for more: https://crc.pitt.edu/h2p
* [Submit a ticket](https://crc.pitt.edu/webform-ticket)

In cluster a lot of these files are in this base directory:
`/ihome/sam/bmooreii/projects/opensoundscape/xeno-canto/`. Anything that needs to be run should be copied to own directory, but anything just sitting there (HD5, output files, label, csvs) can be read from any directory.

Other notes: if any of these models looks really good, we could try to use the script in `continue`, just changing the model and labels directory. We could also predict with the 60-species model: `categorical_inception_v3_only1000.h5`. To get the labels for this, look at
`only1000-400-birds-200K-train-test.csv` - contains the top 60 birds.


### Table of contents:

* [Generate training images](#generate-training-images)
* [Train models](#train-models)
* [Generate prediction images](#generate-prediction-images)
* [Run predictions](#run-predictions)


# Generate training images
We needed to generate spectrograms for the training files first. 

Made a virtual environment:
```
module purge
module load python/3.7.0 venv/wrap
mkvirtualenv preprocess
workon preprocess
pip install pandas numpy scipy librosa pillow
```

Made sure `preprocess.slurm` includes these lines:
```
module load python/3.7.0 venv/wrap
workon preprocess
```

And then edited the `labels.csv` file so that it contains the paths for images and not for mp3s.

# Train models

Every model requires the following:

* `.slurm` file
* `.py` file
* labels file
* A copy of `sigmoid_inception_v3.py`

And may/will output the following:

* `*.out` --> losses. Can read these while they're running & not finished
* `*.h5` --> best, last. Two models, if the models are working correctly. The `best-model.h5` is the one with the highest val_acc. To see accuracies, cat the `.out` file and `grep loss`.


## TensorFlow 1.14.0 model

This model is found in `/ihome/jkitzes/ter38/train-508-birds/restart`. 

### Making virtual environment

It appears we need Python 3.6 to run TF 1.14, because attempting to use the TensorFlow 1.14.0 wheels on the cluster resulted in a "not supported" error. Instead of using the wheels at all, I pip-installed tensorflow-gpu. (This might be the cause of problems, too?)

Created a virtual environment using the following:

```
module purge
module load python/anaconda3.6-5.2.0 venv/wrap # Note the use of 3.6 python for tensorflow 1.14.0
mkvirtualenv tf-1.14.0-gpu #automatically activates venv
pip install tensorflow-gpu==1.14.0 pandas numpy scipy librosa pillow
```

In the future that `module load` line should also include `cuda/10.0.130` (see below).

### Model approaches
* **Model 1**, `restart_titanx`: a model submitted to the titanx partition that sees 50% of the dataset per epoch. This model was stopped after ~20 hours of running without indication that it finished an epoch. Virtual environment was loaded in the slurm script via:

       module purge
       module load python/anaconda3.6-5.2.0 venv/wrap
       workon tf-1.14.0-gpu


* **Model 2**, `restart_v100`: v100 model that sees 50% of the dataset per epoch. Virtual environment was loaded in the slurm script via the following commands, which Barry used in his slurm scripts for the same model:

       module purge
       module load python/3.7.0 venv/wrap cuda/10.0.130
       workon tf-1.14.0-gpu


* **Model 3**, `restart_10pct_v100`: v100 model that sees 10% of the dataset per epoch. Virtual environment was loaded in the slurm script via the same commands as Model 1 above. However, this model never started running; we cancelled it and model 1 because of the below:

We think that the `cuda/10.0.130` is necessary for model training, because Barry loaded it in his model training and we see these warnings in the `out` file of Model 1:

```
$ cat restart_titanx/categorical_inception_v3_1000s_titanx.out  | grep cud
2019-11-25 15:04:02.440364: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcuda.so.1
2019-11-25 15:04:02.781290: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Could not dlopen library 'libcudart.so.10.0'; dlerror: libcudart.so.10.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /ihome/crc/install/python/anaconda3.6-5.2.0/lib
2019-11-25 15:04:02.787460: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Could not dlopen library 'libcudnn.so.7'; dlerror: libcudnn.so.7: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /ihome/crc/install/python/anaconda3.6-5.2.0/lib
```

Compared to Model 2, where the `cuda` bit was included: 
```
$ cat restart_v100/categorical_inception_v3_1000s.out | grep cud
2019-11-26 10:22:54.331202: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcuda.so.1
2019-11-26 10:22:54.842213: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcudart.so.10.0
2019-11-26 10:22:55.352076: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcudnn.so.7
2019-11-26 10:22:55.354461: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcudart.so.10.0
2019-11-26 10:23:19.036679: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcudnn.so.7
```

So we killed Models 1 and 3, and submitted:

* **Model 4**, `restart_10pct_titanx`: a 10%-per-epoch model on titanx, with the following environment loading in the slurm script:

       # The only difference between this and the Model 2 load is the use of python 3.6 instead of 3.7.
       module purge
       module load python/anaconda3.6-5.2.0 venv/wrap cuda/10.0.130
       workon tf-1.14.0-gpu



#### Making these models:
First, I copied from Barry's files the following: Python file, slurm file, a needed dependency (`sigmoid_inception_v3.py`), and the labels file. The following modifications were required to run the file in this location:

* Slurm file:
        * Changing the job name to something descriptive
        * Pointing to the correct Python file, if the Python filename was changed
        * Changing the instructions to load the venv (differs for each model--see section above)
        * Increasing the amount of memory allowed `#SBATCH --mem=100G`
        * `titanx` models (1 and 4) only: changing GPU partition used: `#SBATCH --partition=titanx` 

* Python file:
        * Changing the filename of this file in some cases (e.g. adding `_titanx`)
        * Pointing to the correct path of the labels file
        * Model 4 only: changing the amount of data seen per epoch to 10%
        * `titanx` models (1 and 4) only: changing the batch size (can only use 64 on the `v100`, not the `titanx`): `BATCH_SIZE = 32 * NUM_GPUS`

After this, I submitted the slurm file: `sbatch <filename>.slurm`

### Pytorch model

It appears that we're loading a pytorch environment from `conda`, so we don't have to make a venv for this model. 

We copied the Python and slurm files to the cluster, then made some changes to run them on `titanx`:
* Python file:
       * Fix a bug: `Line 48, in <module>: name 'model' not defined`: change line 48 from `num_ftrs = model.fc.in_features` to `num_ftrs = net.fc.in_features`
       * Fix another bug: the original file didn't call `lr_scheduler` with reference to pytorch model `optim` in line 57; change line 57 to: `exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)`
       * Change batch size in two places to 32 (search for "batch")
* Slurm file:
       * Reduce the number of CPUs per task: `#SBATCH --cpus-per-task=1`
       * Change the partition to one we can use: `#SBATCH --partition=titanx`

Nevermind, there are too many bugs to run this. Most recent bug can be found in `/ihome/jkitzes/ter38/train-508-birds/pytorch/1000s_inceptionv3.out`

# Generate prediction images

Create a virtual environment:

```
module purge
module load python/3.7.0 venv/wrap
mkvirtualenv splitfiles
pip install ray
```

Copy `split_macaulay.py` and `split_macaulay.slurm` to image generation directory

Submit to `smp` cluster, `smp` partition with `150G` of memory and 20 nodes per cpu: `sbatch split_macaulay.slurm`.

# Predict on images

### Create virtual environment
We made a venv using these instructions.
```
module purge
module load python/3.7.0 venv/wrap
mkvirtualenv tf-2.0.0-gpu # or make a different virtualenv for a cpu if you feel like you're waiting too long
pip install /ihome/crc/build/tensorflow/tensorflow-2.0.0/whl_file/<optional subdirectory for cpu version: cpu/>tensorflow-2.0.0-cp37-cp37m-linux_x86_64.whl pandas pillow scipy librosa sklearn
```

To work in this environment, need to be in a GPU machine and need to load the cuda module that corresponds to it. BUT this is already in the slurm script, so 
```
module purge
module load python/3.7.0 venv/wrap cuda/10.0.130
workon tf-2.0.0-gpu
```

Some notes about the settings Barry used:
1 GPU, 6 CPUs, 6 Workers

### Run predictions
Edit the `labels.csv` file which Sam created to transform the labels into the same order that is in this file: `head -n1 /ihome/sam/bmooreii/projects/opensoundscape/xeno-canto/train-test-split/1000s-dataset.csv` - 

Inside `predictions` are two files, `*.py` and `*.slurm`. Copy these into my directory and edit the files as follows:
* PYTHON file:
    * point to model
    * point to RECREATED labels file (see above). Do a test dataset first, then the real one.
    * change the python file to put the predictions in an output file (possibly dump them to a csv). They are currently just stored in a variable. It'll be a numpy array. We think the structure is as follows. Outer list: how many batches were run. First inner list: 32 or 64. Second inner list: each file's predictions, length 508. Can flatten the top two layers. If flattened, should be the same order as the files. Ultimate size will be 4260 (number of files) x 508 (number of classes)
        
* SLURM file:
   * Modify to point to the correct Python file, and change any other things e.g.
   * OPTIONAL: 
         * Can also change the name of the Python file (`eval.py`) and the out file.
         * Can change it to run on a CPU 
              * make new venv as described above
              * delete all three lines 8, 9, 10 in the slurm script
              * change the workon environment to `cpu`

OPTIONAL: Change partition because v100 is slow (look for anything not showing mix on `sinfo -M gpu`)
* Change in slurm file: Change the actual partition `--partition=v100`
* Change in python file: reduce the batch size to 32 (only on v100 can we do 64)

Run `sbatch eval.slurm` in a virtual environment from the directory that the slurm file is in. The predictions will be output from the Python file. Right now, they just go into a Python variable.

Then verify output. Neither the classes nor the filenames are not written in the prediction output. The order of the files will be the same as in the labels file. The order of the classes will be exactly the same order as Barry's labels, which can be found by `head -n1 /ihome/sam/bmooreii/projects/opensoundscape/xeno-canto/train-test-split/1000s-dataset.csv`.  

First run test prediction on just a few files, then modify .py file again and run on entire set.
