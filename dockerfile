FROM nvidia/cuda:11.0.3-cudnn8-devel-centos7

# yhyu13 : Use conda instead of venv. Downlaod conda and add activation script to bashrc
# Using conda in docker file reference: 
# https://stackoverflow.com/questions/65492490/how-to-conda-install-cuda-enabled-pytorch-in-a-docker-container
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# yhyu13 : install additional packages
RUN yum install -y curl

# yhyu13 : donwload anaconda package & install
RUN curl "https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh" -o /anaconda.sh && \
        /bin/bash /anaconda.sh -b -p /opt/conda && \
        rm /anaconda.sh && \
        ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
        echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# yhyu13 : add conda to path  
ENV PATH /opt/conda/bin:/opt/conda/condabin:$PATH

# yhyu13 : create conda env
RUN conda create -n minitorch python=3.9

# yhyu13 : add conda env to enviorment
RUN echo "conda activate minitorch" >> ~/.bashrc
ENV PATH /opt/conda/envs/minitorch/bin:$PATH

# yhyu13 : now everything installed under textgen env by default
RUN source ~/.bashrc && \
    conda activate minitorch && \
    mkdir -p /envs

COPY ./requirements.txt /envs

COPY ./requirements.extra.txt /envs

RUN python -m pip install -r /envs/requirements.txt && \
    python -m pip install -r /envs/requirements.extra.txt && \
    rm -rf /envs

