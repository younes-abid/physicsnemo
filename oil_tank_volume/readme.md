# To run your first example in docker container, you can use the following command:

```bash
cd oil_tank_volume
bash scripts/build_docker.sh
bash scripts/run_docker.sh

```

# To run the Jupyter Notebook server, you can use the following command:
```bash
# inside the container
# the port is binded to 8889 on the host machine
jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
```