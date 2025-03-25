#!/bin/bash

# Step 1: Install dependencies
echo "Installing dependencies..."
if ! pip install -r requirements.txt; then
    echo "Failed to install dependencies. Exiting."
    exit 1
fi

# Step 2: Run data preparation script
echo "Running data_prep.py to generate vectorstore..."
if ! python data_prep.py; then
    echo "Failed to run data_prep.py. Exiting."
    exit 1
fi

# Step 3: Start the Streamlit app
echo "Starting Streamlit app..."
streamlit run rag.py
