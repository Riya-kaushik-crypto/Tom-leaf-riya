#!/bin/bash

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Create necessary directories
mkdir -p model static/uploads dataset/{train,validation}/{early_blight,late_blight,healthy}

# Train the model (optional - if you have dataset)
# python model/train_model.py

# Run the Flask app
python app.py