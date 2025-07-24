FROM mcr.microsoft.com/devcontainers/python:1-3-bookworm

ARG APT_SOURCES_FILE=.devcontainer/debian.sources
ARG PIP_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Set up APT sources to boost installation speed
COPY ${APT_SOURCES_FILE} /tmp/debian.sources
RUN mv /etc/apt/sources.list.d/debian.sources /etc/apt/sources.list.d/debian.sources.bak \
    && mv /tmp/debian.sources /etc/apt/sources.list.d/debian.sources \
    && apt-get update

# Set up pip index URL
RUN pip config set global.index-url ${PIP_INDEX_URL}
