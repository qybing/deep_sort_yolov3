FROM tensorflow/tensorflow:1.10.0-gpu-py3
RUN   apt-get update && apt-get install -yq  libgtk2.0-dev && \
mkdir /root/.pip/ && cd /root/.pip/ && touch pip.conf && echo '[global]' >> pip.conf && \
echo 'index-url = https://pypi.tuna.tsinghua.edu.cn/simple' >> pip.conf && echo 'trusted-host = pypi.tuna.tsinghua.edu.cn' >> pip.conf
COPY . /opt/code/deep_sort_yolov3
WORKDIR /opt/code/deep_sort_yolov3
RUN pip install --upgrade pip && pip3 install keras==2.2.4 && pip3 install -r requirements.txt && \
rm -rf /var/lib/apt/lists/*