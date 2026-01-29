if run at the first time, then create a container:
```
xhost +
docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY --network=host --gpus=all --name=legged_gym_container caozx1110/legged_gym:v1 /bin/bash
```

if the container has been created, then run:
```
xhost +
docker start legged_gym_container
docker exec -it legged_gym_container /bin/bash
```
then in docker container, run:

```
cd ~/legged_gym

python ./legged_gym/scripts/train.py --task=a1 --num_envs=1024 --headless
```
now training starts.

in container, using exit to quit.

to stop the container, run:

```
docker stop legged_gym_container
```

```
docker run -it \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  --network=host \
  --gpus=all \
  --name=legged_gym_container \
  -v /home/finnox-4090/RL/Pikachu:/home/gymuser/pikachu \  
  caozx1110/legged_gym:v1 \
  /bin/bash


pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
  --index-url https://download.pytorch.org/whl/cu118

# 设置本地环境
export PYTHONPATH="/home/gymuser/pikachu/unitree_rl_gym_go2:$PYTHONPATH"
```


```
# 同步网络模型
scp -r finnox-4090@10.10.28.39:~/RL/Pikachu/unitree_rl_gym_go2/logs/Pikachu_V01 /mnt/data/Projects/unitree_rl_gym/logs/

# 同步训练参数
scp -r finnox-4090@10.10.28.39:~/RL/Pikachu/unitree_rl_gym_go2/legged_gym/envs/pikachu /mnt/data/Projects/unitree_rl_gym/legged_gym/envs

scp -r finnox-4090@10.10.28.39:~/RL/Pikachu/unitree_rl_gym_go2/resources/robots/Pikachu_V01 /mnt/data/Projects/unitree_rl_gym/resources/robots/


```


```
# 打包镜像

docker save -o legged_gym_v1.tar caozx1110/legged_gym:v1
```

python play.py --task=Pikachu_V01 --num_envs=10 --load_run=Jan27_12-21-51 --checkpoint=3000

```
 # 重启
# 停止所有docker容器
docker stop $(docker ps -aq)

# 重启docker服务
sudo systemctl restart docker

# 重启NVIDIA服务（最常用方法）
sudo systemctl restart nvidia-persistenced

# 重新启动容器
docker start [容器名]

```