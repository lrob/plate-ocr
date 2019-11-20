FROM nvidia/cuda:10.0-cudnn7-devel as base
RUN apt update

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y \
      git libsm6 libxrender1 libfontconfig1 cmake \
      python3 python3-pip python3-tk libv4l-dev

RUN mkdir -p /mnt
#RUN groupadd -g 1000 sa
#RUN groupadd -g 999 docker
#RUN useradd --system --home-dir /mnt --shell /bin/bash --uid 1000 --gid 999 -G 1000 sa
#RUN chown sa.docker /mnt -R

#USER sa

#ENV PYTHONPATH=${PYTHONPATH}:/mnt/.local/lib/python3.6/site-packages:/mnt/src/lib/python3.6/site-packages:/mnt/src/venv/lib/python3.6/site-packages:/spinretail/research
#ENV PATH=${PATH}:/mnt/.local/bin
RUN python3 -m pip install --upgrade pip

RUN pip3 install --user \
      tensorflow-gpu==2.0.0 matplotlib
      # git+https://github.com/atareao/python3-v4l2capture
      # scipy  pandas dlib pillow scikit-learn scikit-image

ADD preprocessing.py /mnt
WORKDIR /mnt

CMD ["python3", "preprocessing.py"]