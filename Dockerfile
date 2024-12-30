# Use an official Python runtime as a parent image
#FROM python:3.9-slim

# Set the working directory in the container to /app
#WORKDIR /app

# Copy the current directory contents into the container at /app
#COPY . /app

# Install dependencies
#RUN pip install --no-cache-dir -r requirements.txt

# Clean up unnecessary files
#RUN rm -rf /root/.cache

# Make port 80 available to the world outside this container
#EXPOSE 80

# Define the command to run the app using uvicorn
#CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]


# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Nginx
RUN apt-get update && apt-get install -y nginx

# Copy Nginx configuration file
COPY nginx.conf /etc/nginx/nginx.conf

# Expose port 80
EXPOSE 80

# Start Nginx and the FastAPI application
CMD ["sh", "-c", "nginx && uvicorn app:app --host 0.0.0.0 --port 8080"]
