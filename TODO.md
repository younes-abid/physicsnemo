
### **1. Define Additional Metrics**

- Implement functions to compute regression metrics:
    - **Mean Absolute Error (MAE)**:
        
        def compute_mae(predictions, targets):
        
        return torch.mean(torch.abs(predictions - targets))
        
    - **Mean Squared Error (MSE)**:
        
        def compute_mse(predictions, targets):
        
        return torch.mean((predictions - targets) ** 2)
        
    - **R-squared (R²)**:
        
        def compute_r2(predictions, targets):
        
        ss_total = torch.sum((targets - torch.mean(targets)) ** 2)
        
        ss_residual = torch.sum((targets - predictions) ** 2)
        
        return 1 - (ss_residual / ss_total)
        

---

### **2. Compute Metrics During Validation**

- Modify the validation block to compute these metrics:
    - After computing loss_valid, also compute `MAE`, `MSE`, and `R²` using the above functions.
    - Accumulate these metrics over all validation steps.

---

### **3. Log Metrics to TensorBoard**

- Use the SummaryWriter to log the new metrics:
    
    writer.add_scalar("validation_mae", average_mae, cur_nimg)
    
    writer.add_scalar("validation_mse", average_mse, cur_nimg)
    
    writer.add_scalar("validation_r2", average_r2, cur_nimg)
    

---

### **4. Log Metrics to logger0**

- Add the new metrics to the periodic stats log:
    
    logger0.info(f"validation_mae: {average_mae:.4f}")
    
    logger0.info(f"validation_mse: {average_mse:.4f}")
    
    logger0.info(f"validation_r2: {average_r2:.4f}")
    

---

### **5. Update the Training Loop**

- Ensure the training loop is updated to compute and log these metrics periodically.

---

### **6. Test the Implementation**

- Verify that the new metrics are being computed correctly.
- Check that the metrics appear in both TensorBoard and the console logs.

---

### **Deliverables**

1. **Updated Training Code**:
    - Functions to compute MAE, MSE, and R².
    - Integration of these metrics into the validation block.
    - Logging of these metrics to TensorBoard and logger0.
2. **Verification**:
    - Ensure the new metrics are visible in TensorBoard.
    - Confirm the metrics are logged to the console.

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
