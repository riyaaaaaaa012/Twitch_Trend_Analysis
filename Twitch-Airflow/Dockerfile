FROM apache/airflow:2.8.1-python3.11

USER root

# Update and install system dependencies
RUN apt-get update && apt-get install -y \
    git curl gcc g++ \
    && rm -rf /var/lib/apt/lists/*

USER airflow
WORKDIR /opt/airflow

# Copy requirements and install python dependencies
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /requirements.txt

# Install CUDA-enabled PyTorch (2.0.1 with CUDA 11.8)
# RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip install torch==2.0.1+cpu torchvision==0.15.2+cpu torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu


# Download nltk data required
RUN python3 -m nltk.downloader stopwords punkt punkt_tab

# Optional: Copy your DAGs (uncomment and adjust if needed)
# COPY dags/ /opt/airflow/dags/
