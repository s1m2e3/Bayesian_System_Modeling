# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.10
# Install pip requirements
COPY requirements.txt .
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install pyro-ppl[extras]

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

