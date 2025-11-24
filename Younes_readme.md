# Younes contrib
```bash
/docker
/scripts
/notebooks
/Younes_readme.md
```
# To run your first example in docker container, you can use the following command:

```bash
bash scripts/build_docker.sh
bash scripts/run_docker.sh
docker exec -it physiscsnemo_container /bin/bash

## This will build the docker image and run the example script inside the container.
bash scripts/example.sh

bash scripts/train_regression_normal_T2_TSK.sh
bash scripts/train_diffusion_normal_T2_TSK.sh
```

```bash
nohup bash scripts/run_docker.sh && bash scripts/train_diffusion.sh > ./output/output.log 2>&1 &
```

```bash
# using tmux
# list tmux sessions
tmux ls
# if tmux session not created yet:
tmux new -s train
# else
tmux attach -t train
# inside the tmux session
# if docker container not running yet
bash scripts/run_docker.sh
# else
docker exec -it physiscsnemo_container /bin/bash

bash scripts/train_regression_normal_SST_PSFC.sh
# Detach from the session by pressing Ctrl + B, then D

#to kill a tmux session
tmux kill-session -t train
```
# To run the Earth2Studio container, you can use the following command:
The earth2studio image is built from another repository, you can find it here:https://github.com/younes-abid/earth2studio 
```git clone --branch develop https://github.com/younes-abid/earth2studio.git temp_e2s```
```bash scripts/build_docker.sh```

```bash
bash scripts/run_docker_e2s.sh
ssh -L 8889:localhost:8889 younes.abid@10.120.125.144
```

# To learn about this repo check the tutorials under /notebooks/tutorials.

# To run the Jupyter Notebook server, you can use the following command:
```bash
#inside the container
jupyter notebook list
jupyter notebook stop 8888
jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
```

# To run Tensorboard, you can use the following command:

```bash
apt update && apt install lsof && sudo lsof -i :6006 | awk 'NR>1 {print $2}' | xargs kill -9

#["U10_V10", "T2_TSK", "Q2_rain_rate", "SST_PSFC"]
tensorboard --logdir=/app/tensorboard/SST_PSFC/regression --host 0.0.0.0 --port 6006
tensorboard --logdir=/app/tensorboard/SST_PSFC/diffusion --host 0.0.0.0 --port 6006
ssh -L 6006:localhost:6006 younes.abid@10.120.125.144
https://localhost:6006
```

# To run Streamlit, you can use the following command:

```bash
streamlit run notebooks/tutorials/cordiff/weather-pipeline/app.py --server.port 8501 
ssh -L 8501:localhost:8501 younes.abid@10.120.125.144
https://localhost:8501
```

# Related documentation:
 - [../examples/generative/README.md](../examples/generative/README.md)