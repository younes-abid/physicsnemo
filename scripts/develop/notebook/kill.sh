unset JUPYTER_ALLOW_INSECURE_WRITES
jupyter notebook list | awk '/http/ {print $1}' | xargs -I {} jupyter notebook stop {}