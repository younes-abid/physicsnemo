To run your first example in docker container, you can use the following command:

```bash
bash scripts/build_docker.sh
bash scripts/run_docker.sh
## This will build the docker image and run the example script inside the container.
bash scripts/example.sh
```
To run the Earth2Studio container, you can use the following command:
The earth2studio image is built from another repository, you can find it here:https://github.com/younes-abid/earth2studio ```bash scripts/build_docker.sh```

```bash

bash scripts/run_docker_e2s.sh

ssh -L 8889:localhost:8889 younes.abid@10.120.125.144_ds02
```
To learn about this repo check the tutorials under /notebooks/tutorials.
To run the Jupyter Notebook server, you can use the following command:

```bash
#inside the container
jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
```