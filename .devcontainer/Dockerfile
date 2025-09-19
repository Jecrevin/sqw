FROM mcr.microsoft.com/devcontainers/python:1-3-bookworm

ARG APT_SOURCES_FILE_DIR=.devcontainer
ARG APT_SOURCES_FILENAME
ARG PIP_INDEX_URL

# Set up APT sources to boost installation speed
COPY "${APT_SOURCES_FILE_DIR}/${APT_SOURCES_FILENAME}*" .devcontainer/noop.txt /tmp/apt-tmp/
RUN if [ -f /tmp/apt-tmp/${APT_SOURCES_FILENAME} ]; then cp /tmp/apt-tmp/${APT_SOURCES_FILENAME} /etc/apt/sources.list.d/debian.sources; fi \
    && rm -rf /tmp/apt-tmp \
    && apt-get update

# Set up pip index URL
RUN if [ -n "${PIP_INDEX_URL}" ]; then pip config set global.index-url ${PIP_INDEX_URL}; fi
