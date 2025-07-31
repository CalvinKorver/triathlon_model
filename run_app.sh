#!/bin/bash
echo "🏊‍♀️🚴‍♀️🏃‍♀️ Starting Triathlon Stage Classifier..."
echo "Activating FastAI environment and launching Streamlit..."

# Activate conda environment and run streamlit
source ~/miniforge3/etc/profile.d/conda.sh
conda activate fastai
streamlit run app.py

echo "App started! Open your browser to the URL shown above."