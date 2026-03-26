# Dockerfile
FROM python:3.10-slim

# Accept build argument
ARG RUN_ID

# Set environment variable
ENV MODEL_RUN_ID=$RUN_ID

# Set working directory
WORKDIR /app

# Copy requirements if any
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a script to simulate model download
RUN echo '#!/bin/bash\n\
echo "========================================="\n\
echo "Deploying Model with Run ID: ${MODEL_RUN_ID}"\n\
echo "========================================="\n\
echo "Downloading model from MLflow..."\n\
# In production, you would actually download:\n\
# mlflow artifacts download --run-id ${MODEL_RUN_ID}\n\
echo " Model downloaded successfully!"\n\
echo "Container ready for deployment!"\n\
' > /app/download_model.sh && chmod +x /app/download_model.sh

# Command to run when container starts
CMD ["/app/download_model.sh"]
