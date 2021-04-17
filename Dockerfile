ARG CUDAVERSION=10.2

FROM nvidia/cuda:${CUDAVERSION}-base

# User Setup
ARG UID
ARG GID
ARG USER_PASSWORD
RUN adduser --disabled-password --gecos "" container_user
RUN usermod  -u ${UID} container_user
RUN groupmod -g ${GID} container_user
RUN echo container_user:${USER_PASSWORD} | chpasswd
RUN usermod -aG sudo container_user

RUN mkdir /home/src
WORKDIR /home/src
ENV HOME /home/src

RUN apt update

# Global Apt Dependencies
COPY apt_requirements.txt $HOME/apt_requirements.txt
RUN cat apt_requirements.txt | xargs apt install -y
RUN rm apt_requirements.txt

# Update pip3
RUN pip3 install --upgrade pip

# Install wandb and initialize
RUN pip3 install --upgrade wandb
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN wandb login --host=<host> <key>

# Cache pytorch so it doesn't re-download on requirements change
RUN pip3 install torch

# Global Python Dependencies
COPY pip_requirements.txt $HOME/pip_requirements.txt
RUN pip3 install -r pip_requirements.txt
RUN rm pip_requirements.txt

USER container_user
