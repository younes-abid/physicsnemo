check this doc /home/younes.abid/git/physicsnemo/examples/README.md

Now look at how we train our model regression and diffusion
/home/younes.abid/git/physicsnemo/scripts/train/weather/regression/train_regression_normal_Fog_index.sh
/home/younes.abid/git/physicsnemo/scripts/train/weather/diffusion/train_diffusion_normal_Fog_index.sh
So as you can see we have dependencies to /home/younes.abid/git/physicsnemo/examples/weather/corrdiff/train.py
/home/younes.abid/git/physicsnemo/examples/weather/corrdiff/datasets/custom_list_2.py
/home/younes.abid/git/physicsnemo/examples/weather/corrdiff/conf/config_training_custom_regression_normal_Fog_index.yaml
/home/younes.abid/git/physicsnemo/examples/weather/corrdiff/conf/config_training_custom_diffusion_normal_Fog_index.yaml

Now in the here /home/younes.abid/git/physicsnemo/examples/README.md there is a mention about generation
Generation
Once both models are trained, you can use generate.py to create new predictions. The generation process requires:

Required Files:

Trained regression model checkpoint

Trained diffusion model checkpoint

Configuration file conf/config_generate_hrrr_mini.yaml

Execute the generation command:

python generate.py --config-name="config_generate_hrrr_mini.yaml" \
  ++generation.io.res_ckpt_filename=</path/to/diffusion/model> \
  ++generation.io.reg_ckpt_filename=</path/to/regression/model>
The output is saved as a NetCDF4 file containing three groups:

input: The original input data

truth: The ground truth data for comparison

prediction: The CorrDiff model predictions

You can analyze the results using the Python NetCDF4 library or visualization tools of your choice.

You should investigate this code /home/younes.abid/git/physicsnemo/examples/weather/corrdiff/generate.py
Prepare a script for generation under /home/younes.abid/git/physicsnemo/scripts/generate/weather/corrdiff/generate_corrdiff_normal_Fog_index.sh
You should investigate this example of config here /home/younes.abid/git/physicsnemo/examples/weather/corrdiff/conf/config_generate_hrrr_mini.yaml
and create your own config file for Fog_index generation under /home/younes.abid/git/physicsnemo/examples/weather/corrdiff/conf/config_generate_custom_Fog_index.yaml

We have the 2 needed checkpoints under
/home/younes.abid/git/physicsnemo/outputs/checkpoints/Fog_index/checkpoints_regression/*.mdlus
/home/younes.abid/git/physicsnemo/outputs/checkpoints/Fog_index/checkpoints_diffusion/*.mdlus
So the checkpoints paths will to the last checkpoint in the folder

The path to the dataset is under /home/younes.abid/git/physicsnemo/data/custom_data_2/ERA5_WRF_combined_concatenated_432/*.nc and we will use the last data which is /home/younes.abid/git/physicsnemo/data/custom_data_2/ERA5_WRF_combined_concatenated_432/2024-04-30_2024-05-30_21.nc
We should be easily able to pick another data if we want in the script by changinf the name of the data file in the script.
NOTE we will run always the scripts inside the docker container as explained in the Younes_readme.md
Note we also should properly set the output paths inside the sctipt the same way we set it in here /home/younes.abid/git/physicsnemo/scripts/train/weather/diffusion/train_diffusion_normal_Fog_index.sh
So our script should be easily maintainable and we should be able to change the paths of the data and the output checkpoints easily



# Oil tank volume prediction
 - add new step as preprocessing at the beginin of thepipeline (decibels, logscale, ......)
 - add xaray contour and contourf methods to detect edges
 - check circle detection methods
 - check in the SOTA is there a way to decode SAR amplitude into 3d 


# cordiff
    - train diffusion
    - move regression+ diffusion to archived and clean the old archive when finished and get results >=0.8 r2
    - pick temperature and retrain regression then diffusion
### **1. Model Improvements**

- **Increase Model Capacity**:
    - Test with a bigger model than the current "normal" configuration.
    - Add skip connections (e.g., ResNet-style) to help the model learn deeper representations.
- **Improve Non-Linearity**:
    - Experiment with different activation functions:
        - `ReLU`
        - `LeakyReLU`
        - `Swish`
- **Regularization**:
    - Add dropout layers to prevent overfitting and memorization of training data.
    - Add L1 or L2 regularization to the model to penalize large weights.
- **Add Attention Mechanisms**:
    - Incorporate attention layers to focus on the most important parts of the input.

---

### **2. Training Improvements**

- **Loss Function**:
    - Use a weighted loss function if certain ranges of the target variable are more important.
    - For regression tasks, consider using:
        - **Huber Loss**: Reduces sensitivity to outliers.
        - **Log-Cosh Loss**: A smoother alternative to MSE.
- **Learning Rate**:
    - Experiment with different learning rate schedules:
        - Cosine annealing
        - Cyclical learning rates
    - If the model is not converging well:
        - Reduce the learning rate.
        - Use a learning rate finder to identify the optimal learning rate.
- **Optimizers**:
    - Experiment with different optimizers:
        - `AdamW`
        - `RMSprop`
        - `SGD with momentum`
- **Regularization During Training**:
    - Add weight decay (L2 regularization) to prevent overfitting.
- **Stabilize Training**:
    - Clip gradients to prevent exploding gradients and stabilize training.
    - If using a very deep network, consider adding batch normalization to stabilize and accelerate training.