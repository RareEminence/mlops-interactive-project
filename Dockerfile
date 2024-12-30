# Use lightweight Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy project files
COPY app/iris_model.pkl /app/iris_model.pkl
COPY requirements.txt /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit's default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "interactive_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
