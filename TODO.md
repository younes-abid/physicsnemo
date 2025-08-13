
## Step 2: Generate NetCDF Files or Packages
- [ ] Identify the script or utility to generate NetCDF files:
  - [ ] Check for `NetCDFWriter` in `physicsnemo.utils.corrdiff.utils`.
  - [ ] Explore scripts like `generate.py` in `examples/weather/corrdiff`.
- [ ] Use the diffusion checkpoint to generate NetCDF files:
  - [ ] Write a script or notebook to generate NetCDF files using the diffusion model.

---

## Step 3: Upload the Generated Files to Earth2Studio
- [ ] Use Earth2Studio's upload tools or APIs to upload the generated NetCDF files.
- [ ] Verify that the files are accessible in Earth2Studio.

---

## Step 4: Run Inference Using Earth2Studio
- [ ] Use Earth2Studio's inference tools to predict on the uploaded data.
- [ ] Modify the inference script if necessary to use the custom dataset and model.

---

## Step 5: Visualize and Analyze Results
- [ ] Visualize the predictions using tools like `matplotlib` or Earth2Studio's visualization tools.
- [ ] Compare the predictions with the ground truth to evaluate the model's performance.

---

Let me know if you'd like to start with a specific step, and we can work on it together!```<!-- filepath: /home/younes.abid/git/physicsnemo/TODO.md -->

# TODO List for Earth2Studio Inference Workflow

## Step 1: Verify and Prepare the Environment
- [ ] Clone the Earth2Studio repository:
  - [ ] `git clone --branch develop https://github.com/younes-abid/earth2studio.git temp_e2s`
- [ ] Build the Earth2Studio Docker image:
  - [ ] `bash scripts/build_docker.sh`
- [ ] Run the Earth2Studio Docker container:
  - [ ] `bash scripts/run_docker_e2s.sh`
- [ ] Start the Jupyter Notebook server inside the container:
  - [ ] `jupyter notebook --ip=0.0.0.0 --no-browser --allow-root`

---

## Step 2: Generate NetCDF Files or Packages
- [ ] Identify the script or utility to generate NetCDF files:
  - [ ] Check for `NetCDFWriter` in `physicsnemo.utils.corrdiff.utils`.
  - [ ] Explore scripts like `generate.py` in `examples/weather/corrdiff`.
- [ ] Use the diffusion checkpoint to generate NetCDF files:
  - [ ] Write a script or notebook to generate NetCDF files using the diffusion model.

---

## Step 3: Upload the Generated Files to Earth2Studio
- [ ] Use Earth2Studio's upload tools or APIs to upload the generated NetCDF files.
- [ ] Verify that the files are accessible in Earth2Studio.

---

## Step 4: Run Inference Using Earth2Studio
- [ ] Use Earth2Studio's inference tools to predict on the uploaded data.
- [ ] Modify the inference script if necessary to use the custom dataset and model.

---

## Step 5: Visualize and Analyze Results
- [ ] Visualize the predictions using tools like `matplotlib` or Earth2Studio's visualization tools.
- [ ] Compare the predictions with the ground truth to evaluate the model's performance.
