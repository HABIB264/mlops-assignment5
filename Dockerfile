# Dockerfile
FROM python:3.10-slim

# Accept build argument
ARG RUN_ID

# Set environment variable
ENV MODEL_RUN_ID=$RUN_ID

# Set working directory
WORKDIR /app

# Copy requirements if you have them
COPY requirements.txt .
RUN pip install --no-cache-dir mlflow scikit-learn numpy pandas

# Create deployment script
RUN echo '#!/bin/bash\n\
echo "========================================="\n\
echo "🐳 DEPLOYING MODEL TO PRODUCTION" \n\
echo "========================================="\n\
echo "Model Run ID: ${MODEL_RUN_ID}"\n\
echo ""\n\
echo "Downloading model artifacts from MLflow..."\n\
# Simulate model download\n\
sleep 2\n\
echo "✅ Model downloaded successfully!"\n\
echo ""\n\
echo "Container is ready for production!"\n\
echo "========================================="\n\
' > /app/deploy.sh && chmod +x /app/deploy.sh

# Command to run when container starts
CMD ["/app/deploy.sh"]
