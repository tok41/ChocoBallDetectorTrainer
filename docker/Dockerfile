FROM chainer/chainer:latest-python3

RUN echo "alias ls='ls --color=auto'" >> /root/.bashrc
RUN echo "alias ll='ls --color=auto -l'" >> /root/.bashrc
RUN ln -s /usr/bin/python3.6 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

WORKDIR /workspace

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y \
	git \
	wget
RUN curl -sL https://deb.nodesource.com/setup_14.x | bash -
RUN apt-get install -y nodejs

COPY requirements.txt /tmp/
RUN pip install --upgrade pip
RUN pip install -r /tmp/requirements.txt
ENV PYTHONPATH /workspace:$PYTHONPATH

RUN jupyter serverextension enable --py jupyterlab
