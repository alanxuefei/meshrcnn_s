# Use NVIDIA CUDA image as the base
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

# Remove the old GPG key
RUN apt-key del 7

# Set timezone to avoid interactive prompt during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y python3-pip git 

# Install PyTorch3D
RUN pip3 install fvcore iopath
RUN pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu117_pyt200/download.html

# Install Detectron2
RUN pip3 install 'git+https://github.com/facebookresearch/detectron2.git'

# Clone Mesh R-CNN repository and install its dependencies
RUN git clone https://github.com/facebookresearch/meshrcnn.git && \
    cd meshrcnn && \
    pip3 install fvcore iopath && \
    python3 setup.py install

# Install JupyterLab
RUN pip3 install jupyterlab
	
# Install Plotly and JupyterLab extension
RUN pip3 install plotly 

# Expose the port JupyterLab will run on·
EXPOSE 8888

# Update package lists and install system packages
RUN apt-get update \
    && apt-get install -y nodejs npm libgl1-mesa-glx \
    && apt-get install -y wget \ 
	&& pip3 install matplotlib scikit-image imageio opencv-python\
    && rm -rf /var/lib/apt/lists/*

RUN pip install ipywidgets
RUN pip install --upgrade jupyter

# Start JupyterLab when the container is run
CMD ["jupyter", "lab", "--allow-root", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''"]