# GomoNova 训练/对弈镜像。
#
# 在 NVIDIA PyTorch 基础镜像上固化全部依赖，并把 gomonova 装成 editable，
# 使 `import gomonova` 不再依赖 cwd / PYTHONPATH。开发机容器重建后直接用本
# 镜像，无需再手动 pip install——此前容器重建丢失手动安装的依赖，曾导致训练
# （ModuleNotFoundError: gomonova）与 Web（ModuleNotFoundError: fastapi）双双崩溃。
#
# 构建： docker build -t gomonova:latest .
# 运行： docker run --gpus all --ipc=host -p 8000:8000 \
#            -v /tmp/gomonova/checkpoints:/workspace/gomonova/checkpoints \
#            gomonova:latest python -m gomonova.web.server --port 8000
FROM nvcr.io/nvidia/pytorch:26.06-py3

WORKDIR /workspace/gomonova

# 先装依赖（利用层缓存：代码变动不触发重装）
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# 再拷代码并 editable 安装
COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 8000
