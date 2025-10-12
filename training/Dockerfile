FROM python:3.13-slim

# Set the working directory for the application
WORKDIR /app


# Copy the dependency file(s) into the container
# This step is done before copying the rest of the code to leverage Docker's build cache
COPY requirements.txt .

# Install dependencies (using 'pip' in the base image)
# '--no-cache-dir' helps keep the image size small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY app.py .

# Expose a port if your application is a web service (e.g., Flask, Django)
# If it's just a script, you can omit this
# EXPOSE 8000

# Command to run the Python application
# Replace 'your_script.py' with the name of your main Python file
CMD ["python", "app.py"]