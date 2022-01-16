# Personal README note.

## Installation w/ conda

```
conda create --name iql-official
conda install python==3.8
conda install -c conda-forge cudnn==8.1.0.77
conda install -c conda-forge cudatoolkit-dev==11.2.2
pip install -r requirements.txt
pip install "jax[cuda111]<=0.21.1" -f https://storage.googleapis.com/jax-releases/jax_releases.html
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
pip install ipython
```

## Installation for robomimic

```
conda install pytorch==1.6.0 torchvision==0.7.0 cpuonly -c pytorch
pip install robomimic==0.2.0
pip install git+ssh://git@github.com/ARISE-Initiative/robosuite.git@offline_study
```

## Training

```
python train_offline.py --env_name=antmaze-large-play-v0 --config=configs/antmaze_config.py --eval_episodes=100 --eval_interval=250000
python train_offline.py --env_name=lift-low-mg-v0 --config=configs/robomimic_config.py --eval_episodes=100 --eval_interval=250000 --save_dir ./tmp/lift-low-mg-v0
```
