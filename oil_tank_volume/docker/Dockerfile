FROM  --platform=linux/amd64 continuumio/miniconda3:22.11.1

# Install system dependencies
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        wget \
        bzip2 \
        libgeos-dev \
        gdal-bin \
        libgdal-dev \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1 \
        tree \
        ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set conda directory variable
ENV CONDA_DIR=/opt/conda

# Set environment variables for GDAL
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal \
    GDAL_DATA=/usr/share/gdal 

# Install Python dependencies
COPY docker/requirements.txt .
RUN pip install --default-timeout=100 -r requirements.txt

# Set environment variables for PROJ
RUN PROJ_LIB=$(python -c "import pyproj; print(pyproj.datadir.get_data_dir())") && \
    echo "export PROJ_LIB=$PROJ_LIB" >> /etc/profile.d/projlib.sh

ENV CPL_LOG=ERROR
WORKDIR /app/

COPY . .
