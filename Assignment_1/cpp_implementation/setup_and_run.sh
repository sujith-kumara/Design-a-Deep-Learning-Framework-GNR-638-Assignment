#!/bin/bash

echo "========================================"
echo "   Deep Learning Framework Setup "
echo "========================================"
echo ""
echo "Hi! I can help you run the training."
echo ""
echo "Which dataset would you like to train on?"
echo "1) Dataset 1 (data_1)"
echo "2) Dataset 2 (data_2)"
echo ""
read -p "Enter your choice (1 or 2): " choice

if [ "$choice" == "1" ]; then
    echo "Great! Starting training on Dataset 1..."
    ./scripts/run_data_1.sh
elif [ "$choice" == "2" ]; then
    echo "Awesome! Starting training on Dataset 2..."
    ./scripts/run_data_2.sh
else
    echo "I didn't understand that. Please run me again and pick 1 or 2."
fi
