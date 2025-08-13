### **Step 1: Verify and Prepare the Environment**

- <input disabled="" type="checkbox"> Ensure the Earth2Studio repository is cloned:
    
    git clone --branch develop https://github.com/younes-abid/earth2studio.git temp_e2s
    
- <input disabled="" type="checkbox"> Build and run the Earth2Studio Docker container:
    
    bash scripts/build_docker.sh
    
    bash scripts/run_docker_e2s.sh
    
- <input disabled="" type="checkbox"> Start the Jupyter Notebook server inside the container:
    
    jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
    

---

### **Step 2: Generate NetCDF Files or Packages**

- <input disabled="" type="checkbox"> Identify the script or utility to generate NetCDF files. Based on the codebase, this might involve:
    - Using NetCDFWriter from physicsnemo.utils.corrdiff.utils.
    - Running a script like generate.py in corrdiff.
- <input disabled="" type="checkbox"> Use the diffusion checkpoint to generate the NetCDF files:
    
    from physicsnemo.utils.corrdiff import NetCDFWriter
    
    # Example: Generate NetCDF file
    
    writer = NetCDFWriter(
    
    f=output_file,
    
    lat=latitude_array,
    
    lon=longitude_array,
    
    input_channels=input_channels,
    
    output_channels=output_channels,
    
    )
    
    writer.write_prediction(channel_name, time_index, ensemble_index, prediction_data)
    

---

### **Step 3: Upload the Generated Files to Earth2Studio**

- <input disabled="" type="checkbox"> Use Earth2Studio's upload tools or APIs to upload the generated NetCDF files or packages.
- <input disabled="" type="checkbox"> Verify that the files are accessible in Earth2Studio.

---

### **Step 4: Run Inference Using Earth2Studio**

- <input disabled="" type="checkbox"> Use Earth2Studio's inference tools to run predictions on the uploaded data.
- <input disabled="" type="checkbox"> If required, modify the inference script to use your custom dataset and model.

---

### **Step 5: Visualize and Analyze Results**

- <input disabled="" type="checkbox"> Visualize the predictions using tools like matplotlib or Earth2Studio's built-in visualization tools.
- <input disabled="" type="checkbox"> Compare the predictions with the ground truth to evaluate the model's performance.