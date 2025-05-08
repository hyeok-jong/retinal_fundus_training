FROM nvcr.io/nvidia/pytorch:23.10-py3

RUN rm /bin/sh && ln -s /bin/bash /bin/sh
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
         apt-transport-https \
         build-essential \
         ca-certificates \
         libssl-dev \
         git \
         curl \
         zip \
         unzip \
         bzip2 \
         htop \
         fonts-powerline \
         software-properties-common \
         tmux \
         cloc \
         nodejs \
         npm \
     && rm -rf /var/lib/apt/lists/*


RUN npm install -g n
RUN n lts

RUN apt-get update
RUN apt-get install -y --no-install-recommends software-properties-common
RUN apt-add-repository multiverse
RUN apt-get update


RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections
RUN apt-get install -y --no-install-recommends fontconfig ttf-mscorefonts-installer
RUN fc-cache -f -v


WORKDIR /tmp
RUN curl -LO https://github.com/source-foundry/Hack/releases/download/v3.003/Hack-v3.003-ttf.zip && \
        unzip ./Hack-v3.003-ttf.zip && \
        cp -r ./ttf /usr/share/fonts/truetype/Hack-font && \
        chmod 644 /usr/share/fonts/truetype/Hack-font/* && \
        fc-cache -f


RUN apt-get install -y zsh
RUN curl -L https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh | sh
RUN chsh -s $(which zsh)
CMD [ "/bin/zsh" ]


RUN apt-get update
RUN pip install --upgrade pip
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt


RUN ldconfig && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/*

RUN apt-get update && apt-get install -y --no-install-recommends openssh-server
RUN mkdir /var/run/sshd
RUN echo 'PASSWORD HERE!' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/#PermitRootLogin yes/PermitRootLogin yes/' /etc/ssh/sshd_config
EXPOSE 8887

WORKDIR /home/intern_lhj

CMD ["/usr/sbin/sshd", "-D"]
