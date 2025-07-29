# SMS Spam Detection with TensorFlow

This project demonstrates how to build a simple SMS spam classifier using TensorFlow and the popular SMS Spam Collection dataset.

## Setup

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the training script:
   ```bash
   python sms_spam_detection_tf/main.py
   ```
   The script will automatically download the dataset, train a model, evaluate it, and save the trained model to `spam_model`.

The dataset comes from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) and is fetched directly from GitHub when running the training script.
