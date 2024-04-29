# Base images 基础镜像
FROM mirrors.tencent.com/todacc/venus-std-base-image:8.0.0

#MAINTAINER 维护者信息
LABEL MAINTAINER="devancao"

#ENV 设置环境变量
ENV GENERIC_REPO_URL http://mirrors.tencent.com/repository/generic/venus_repo/image_res
ENV GENERIC_CLIENT_URL http://mirrors.tencent.com/repository/generic/venus_repo/client

#ENV 设置环境变量
USER root
RUN mkdir -p /data/home  \
    && cd /data/home  \
    && wget -q $GENERIC_CLIENT_URL/hadoop-venus-1.0.tar.gz \
    && tar -xzvf hadoop-venus-1.0.tar.gz\
    && rm -rf hadoop-venus-1.0.tar.gz\
    && cd /data/ \
    && ln -s miniconda3/envs/env-3.7.7 anaconda3 \
    && /data/anaconda3/bin/pip install venus_mdfs

RUN rpm --rebuilddb && yum -y install cmake llvm9.0-devel 
RUN rpm --rebuilddb && yum -y install tlinux-release-scl scl-utils 
RUN rpm --rebuilddb && yum -y install devtoolset-7-gcc devtoolset-7-gcc-c++
RUN export PATH=/usr/lib64/llvm9.0/bin:$PATH
RUN source /opt/rh/devtoolset-7/enable &&  source ~/.bashrc && conda deactivate && conda activate env-3.7.7 
RUN mkdir /data/tmp \
    && cd /data/tmp \
    && wget https://mirrors.tencent.com/repository/generic/venus_repo/plato_res/common_tools/openmpi-4.1.1.tar.gz  \
    && tar -zxvf openmpi-4.1.1.tar.gz \
    && cd openmpi-4.1.1/ \
    && ./configure --prefix=/usr/local/openmpi && make -j 48 && make install -j 48 \
    && cd .. \
    && rm openmpi-4.1.1.tar.gz

RUN mkdir -p /data/tmp \
    && cd /data/tmp \
    && wget -q http://mirrors.tencent.com/repository/generic/venus_repo/image_res/cuda11.1/cuda_11.1.0_455.23.05_linux.run \
    && wget -q http://mirrors.tencent.com/repository/generic/venus_repo/image_res/cuda11.1/cudnn-11.3-linux-x64-v8.2.0.53.tgz \
    && wget -q http://mirrors.tencent.com/repository/generic/venus_repo/image_res/cuda11.1/cuda11.1.bashrc\
    && chmod 755 cuda_11.1.0_455.23.05_linux.run \
    && ./cuda_11.1.0_455.23.05_linux.run --toolkit --silent  --samples \
    && ldconfig \
    && tar xf /data/tmp/cudnn-11.3-linux-x64-v8.2.0.53.tgz -C /usr/local \
    && chmod a+r /usr/local/cuda/lib64/libcudnn* \
    && ln -s /usr/local/cuda/lib64/libcusolver.so.11  /usr/local/cuda/lib64/libcusolver.so.10 \
    && cp /data/tmp/cuda11.1.bashrc ~mqq/ \
    && cp /data/tmp/cuda11.1.bashrc ~/ \
    && chown -R mqq:mqq ~mqq \
    && wget -P /tmp $GENERIC_REPO_URL/cpu/clean-layer.sh \
    && sh /tmp/clean-layer.sh

RUN cd /usr/local \
    && wget https://mirrors.tencent.com/repository/generic/venus_repo/image_res/cuda11.1/nccl_2.8.4-1+cuda11.1_x86_64.txz \
    && tar xvf nccl_2.8.4-1+cuda11.1_x86_64.txz \
    && cd  nccl_2.8.4-1+cuda11.1_x86_64 \
    && cd include  \
    && cp -r * /usr/local/include \
    && cd ../lib \
    && cp -r * /usr/local/lib \
    && cp -r * /usr/local/lib64 \
    && cd ../.. \
    && rm nccl_2.8.4-1+cuda11.1_x86_64.txz \
    && rm -rf nccl_2.8.4-1+cuda11.1_x86_64

RUN source /opt/rh/devtoolset-7/enable &&  source ~/.bashrc  \
    && conda deactivate && conda activate env-3.7.7 \
    && pip install torch==1.13.1+cu113 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 \
    -f https://download.pytorch.org/whl/torch_stable.html \
    && pip install sentencepiece && pip install nltk && pip  install pybind11 && pip install boto3

ENV  PATH=$PATH:/usr/local/openmpi/bin:/usr/sbin:/usr/local/app/.nvm/versions/node/v8.16.2/bin:/usr/local/app/.pyenv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/data/anaconda3/bin:/usr/local/jdk/bin:/usr/local/sbin:/usr/local/bin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/openmpi/lib:/usr/local/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/data/anaconda3/lib64:/usr/local/lib64:/usr/local/lib/openmpi:/usr/local/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/data/anaconda3/lib64:/usr/local/lib64


#RUN 执行以下命令
USER mqq
RUN mkdir -p /data/install \
    && cd /data/install && wget https://mirrors.tencent.com/repository/generic/orlando_test/deepspeed/ssh5.zip \
    && unzip ssh5.zip && rm ssh5.zip && mkdir -p ~/.ssh && chmod 700 ~/.ssh \
    && cp ./authorized_keys ~/.ssh && cp ./id_rsa* ~/.ssh && chmod go-rwx ~/.ssh/au* ~/.ssh/id*

USER root
RUN cd /data/install && mkdir -p ~/.ssh && chmod 700 ~/.ssh \
    && cp ./authorized_keys ~/.ssh && cp ./id_rsa* ~/.ssh && chmod go-rwx ~/.ssh/au* ~/.ssh/id* \
    && cp -f ssh_host* /etc/ssh && cp -f ./mpi_sshd_config /etc/ssh/ \
    && cd /etc/ssh && chown root:root ./ssh_host* \
    && chmod go-rwx ./ssh_host* && chmod go+r ./*.pub && echo -e "\tStrictHostKeyChecking no" >> /etc/ssh/ssh_config \
    && echo -e "\tUserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config

USER mqq
RUN cd ~ \
    && wget https://mirrors.tencent.com/repository/generic/venus_repo/plato_res/common_tools/openmpi.bashrc \
    && mv openmpi.bashrc ~/.bash_profile \
    && source ~/.bash_profile \
    && source ~/.bashrc \
    && source /opt/rh/devtoolset-7/enable \
    && conda deactivate && conda activate env-3.7.7 \
    && echo -e '\
    export PATH="$PATH:/usr/local/openmpi/bin:/usr/sbin:/usr/local/app/.nvm/versions/node/v8.16.2/bin:/usr/local/app/.pyenv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/data/anaconda3/bin:/usr/local/jdk/bin:/usr/local/sbin:/usr/local/bin:/usr/bin:/sbin:/bin"' >>~/custom.bashrc \
    && pip install cython protobuf PyYAML aiohttp netifaces portalocker \
    && pip install cmake \
    && gcc --version \
    && HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 HOROVOD_WITH_MPI=1 HOROVOD_WITHOUT_GLOO=1 pip install --no-cache-dir horovod
ENV  PATH=$PATH:/usr/local/openmpi/bin:/usr/sbin:/usr/local/app/.nvm/versions/node/v8.16.2/bin:/usr/local/app/.pyenv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/data/anaconda3/bin:/usr/local/jdk/bin:/usr/local/sbin:/usr/local/bin:/usr/bin:/sbin:/bin

RUN  source /opt/rh/devtoolset-7/enable \
    && source ~/.bashrc \
    && source ~/custom.bashrc \
    && conda deactivate && conda activate env-3.7.7 \
    && pip install transformers==4.23.1 \
    && pip install tqdm \
    && pip install accelerate \
    && pip install datasets==1.17.0 \
    && pip install argparse \
    && pip install tensorboard==2.10.0 \
    && pip install rouge_score \
    && pip install deepspeed==0.10.0 \
    && pip install path \
    && pip install peft \
    && pip install wandb==0.12.10 \
    && pip install torchtyping \
    && pip install sentencepiece==0.1.96 \
    && pip install fairscale==0.4.5 \
    && pip install ipython \
    && pip install protobuf==3.19.0 \
    && python -c "import nltk; nltk.download('punkt', quiet=True)"


# 环境变量
ENV OMP_NUM_THREADS 16
ENV PATH /opt/rh/devtoolset-7/root/usr/bin/:/data/miniconda3/bin/:$PATH
ENV LD_LIBRARY_PATH /data/miniconda3/lib/:/usr/local/lib:/lib:/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV LANG "en_US.UTF-8"

# 安装完成后，切换root用户，执行清理垃圾脚本
USER root
RUN wget -P /tmp http://mirrors.tencent.com/repository/generic/venus_repo/image_res/cpu/clean-layer.sh \
    && sh /tmp/clean-layer.sh

WORKDIR /usr/local/app
EXPOSE 80