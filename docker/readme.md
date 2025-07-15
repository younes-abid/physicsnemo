To run your first example in docker container, you can use the following command:

```bash
bash scripts/build_docker.sh
bash scripts/run_docker.sh
## This will build the docker image and run the example script inside the container.
bash scripts/example.sh
```

To learn about this repo check the tutorials under /notebooks/tutorials.
To run the Jupyter Notebook server, you can use the following command:

```bash
#inside the container
jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
```