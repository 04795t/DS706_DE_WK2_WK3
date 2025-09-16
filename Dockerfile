# Use the official Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install pandas numpy matplotlib scikit-learn

# Copy fles
COPY . .

# Run the test script
CMD ["python", "test_happiness.py"]
