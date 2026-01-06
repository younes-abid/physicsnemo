I have created a notebook in here to prepare and train the data check it /home/younes.abid/git/physicsnemo/notebooks/tutorials/cordiff/2-training.ipynb
I can tell that the structure of the data to train cordiff is as follows:
it is netcdf files with 2 groups: "input" and "output"
it has the following 
input_group = xr.open_dataset(output_file, group="input")
input_group
xarray.Dataset

Dimensions:
sample: 3600y_lr: 432x_lr: 432

Coordinates: (0)

Data variables: (17)

Indexes: (0)

Attributes: (0)
output_group = xr.open_dataset(output_file, group="output")
output_group
xarray.Dataset

Dimensions:
sample: 3600y_hr: 432x_hr: 432

Coordinates: (0)

Data variables: (11)

Indexes: (0)

Attributes: (0)

I have trained the cordiff model using these two scripts
/home/younes.abid/git/physicsnemo/scripts/train_regression_normal_Fog_index.sh
/home/younes.abid/git/physicsnemo/scripts/train_diffusion_normal_Fog_index.sh

Now I want to adapt some data to be used with cordiff.
the data is SAR to height data. here /home/younes.abid/git/physicsnemo/data/sar2height
The data has as input ICEYE amplitude values and as output DSM values.
The raw data is in tiff format.
for example:
/home/younes.abid/git/physicsnemo/data/sar2height/run2/ICEYE_X2_SLC_SLH_56495_20210517T180931_Intensity_DB.tif
/home/younes.abid/git/physicsnemo/data/sar2height/run2/ICEYE_X2_SLC_SLH_56495_20210517T180931_DSM_RADAR.tif


We need to create a proper notebook and call it 5-sar2height.ipynb and put it under /home/younes.abid/git/physicsnemo/notebooks/tutorials/cordiff/
The notebook will prepare the data to be used with cordiff model.
The steps are as follows:
1- read the tiff files using rasterio (there are multiple files for input and output)
2- take one input _Intensity_DB.tif and its corresponding output _DSM_RADAR.tif one by one and do the following
    a- properly coregister the input and output if needed (resample, crop, align, etc)
    b- get the AOI of this data 
    c- from _Intensity_DB.tif create the derivated features to enrich the input data as follows:
        i. Original intensity values (dB)
        ii. Percentile rescaled (2-98th percentile)  
        iii. Linear intensity (10^(dB/10))
        iv. Square root of linear intensity
        v. Power transform (α=0.3)
        vi. Speckle-filtered intensity (Lee filter)
        vii. Local mean (5x5)
        viii. Local std (5x5) 
        ix. Sobel magnitude and direction
        x. GLCM contrast and homogeneity
        xi. Multi-scale mean (3x3, 7x7)
        xii. Gradient magnitude
        xiii. Local incidence angle (if available)

        Note: Since your files are named _Intensity_DB.tif, the incidence angle might be embedded in the TIFF metadata or as a separate band. We should check this during implementation.
    
    d- stack all these features to create a multi-channel input data (the number of channels will be equal to the number of features created in step c)
    e- create patches of size 432x432 for input and output (the dimension of patches is a parameter that can be changed later) by default use 432x432 with a stride of 0 (i.e no overlap between patches)
    f- go to /home/younes.abid/git/physicsnemo/data/sar2height/patches and 
        i- get all the existing netcdf to get all AOI that are already used
        ii- make sure that the new patches do not overlap with existing AOI (if they do overlap, skip them)
    g- save only the non overlapping patches with other AOI in netcdf format with two groups "input" and "output" like shown above in one netcdf file. Note the file should contain multiple samples (patches) in the sample dimension. and should contain the AOI information in the attributes of the netcdf file. the overlapped and skipped patches should be saved in a separate netcdf file for record keeping under /home/younes.abid/git/physicsnemo/data/sar2height/patches_skipped
    h- ensure that the saved netcdf files are properly named and organized for easy access during training.
3- repeat step 2 for all the input/output tiff files
4- finally we will have multiple netcdf files that can be used to train cordiff model (similar to the ones used in 2-training.ipynb)
5- compute the statistics of the input and output data (mean and std) to be used for normalization during training (similarly to what is done in 2-training.ipynb)

Note always refers to the existing notebook 2-training.ipynb for reference and code snippets that can be reused in the new notebook. The idea is to have a similar structure and format as much as possible, and have similar netcdf files for training with cordiff that follows the same structure as the existing ones. Additionally, ensure that the new notebook includes detailed comments and explanations for each step to facilitate understanding and future modifications. Also, consider adding a section for error handling and data validation to ensure robustness in the data preparation process. Ensure that the new notebook is well-organized and follows best practices for coding and documentation. It will be very helpful if you include a small section at the beginning to investigate the data and visualize some samples of input and output to better understand the data before proceeding with the preparation steps. Also consider adding visualizations of the derived features to illustrate their characteristics and relevance to the task at hand. Also at the beginning I would love to see some statistics of the raw data (mean, std, min, max, histogram, etc) for both input and output data. Also see AOI coverage map if possible with overlaps and the name of the areas of interest taken from the .tif name. Additionally, include a section to document the methodology used for data preparation and any assumptions made during the process to ensure clarity and reproducibility.

As this is a big task, we can break it down into multiple steps and I will ask you to help me with each step one by one.
first step is to get familiar with the context and check the existing notebook 2-training.ipynb and the data provided in /home/younes.abid/git/physicsnemo/data/sar2height then create a folder under /home/younes.abid/git/physicsnemo/notebooks/tutorials/cordiff/ called 5-sar2height and in that folder create first as many python scripts as needed to perform the tasks mentioned above in step 2 (a to h).
Once the scripts are ready we can then create the notebook that uses these scripts to perform the data preparation.
Do not start writing the notebook yet, just focus on creating the necessary scripts first. The notebook will be smaller and easier to create once the scripts are ready. create now the scripts needed when you finish let me know the names of the scripts created and a brief description of what each script does (use doc and comments in the scripts themselves).

later I will ask you to help me write the notebook that uses these scripts to perform the data preparation.


-------------
ok now we have all the scripts ready I want you to create the notebook 5-sar2height.ipynb under /home/younes.abid/git/physicsnemo/notebooks/tutorials/cordiff/5-sar2height 

the note book should import and use all the scripts and should be nicely structured as high level that make good use of other scripts then provide extra vizualization

Now for vizualization I think the code will be long so it is good idea to first create another scrits on or many for visualization purposes first to keep the long code there