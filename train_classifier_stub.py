# train_classifier_stub.py
"""
Skeleton for training an action classifier.
Collect pose features (time windows) -> label (active/idle) -> train an LSTM or 1D-CNN.
"""

import numpy as np
import pandas as pd

def extract_features_from_video(video_path):
    # implement: read video, extract pose landmarks per frame, create sliding windows
    pass

def train_model(X, y):
    # implement: build and train model using tensorflow/keras or PyTorch
    pass

def main():
    # steps:
    # 1) Collect labelled clips (idle vs active)
    # 2) Extract features / windows
    # 3) Train & save model
    pass

if __name__ == "__main__":
    main()
