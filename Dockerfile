# For more information, please refer to https://aka.ms/vscode-docker-python
FROM ubuntu:latest
RUN apt-get update
RUN apt-get install python3.10
RUN apt-get install -y python3-pip
# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIPX_HOME=/opt/pipx
ENV PATH=/opt/pipx/bin:$PATH
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install --upgrade pip
WORKDIR /app
COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug

CMD ["sh", "-c", "while true; do sleep 10000; done"]

