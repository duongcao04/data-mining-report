#!/bin/bash
echo "========================================"
echo "Khoi dong Customer Churn Prediction API"
echo "========================================"
echo ""

# Kich hoat virtual environment
source venv/bin/activate

# Khoi dong API
echo "Dang khoi dong API tai http://127.0.0.1:8000"
echo ""
uvicorn demo.app:app --reload

