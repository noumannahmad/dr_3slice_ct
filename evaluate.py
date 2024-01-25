import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time
import sys
import os
import glob
import torch.nn.functional as F
import shutil

def evaluate(net, loader, B, N):
    """
    This function evaluates a neural network model on a dataset using batch processing.

    Args:
    - net: The neural network model to be evaluated.
    - loader: A data loader providing input data and ground truth in batches.
    - B: Batch size used for processing data.
    - N: Total number of data samples.

    Returns:
    - A tuple containing:
      - values_out_means: Estimated means for the target values.
      - values_out_vars: Estimated variances for the target values.
      - values_gt: Ground truth values for the target values.
    """

    # Move the neural network model to the GPU and set it to evaluation mode
    net = net.cuda()
    net.eval()

    # Initialize arrays to store results
    values_out_means = []  # Estimated means
    values_out_vars = []   # Estimated variances
    values_gt = []         # Ground truth values

    i_start = 0  # Starting index for the current batch
    T = None     # Number of target values

    # Loop through the data loader
    for X, Y in loader:

        # Lazy initialization of T (number of target values)
        if T is None:
            T = Y.size(1)  # Extract the number of target values
            # Initialize empty arrays based on the number of samples (N) and targets (T)
            values_out_means = np.zeros((N, T))
            values_out_vars = np.zeros((N, T))
            values_gt = np.zeros((N, T))

        # Move input data to the GPU
        X = X.cuda(non_blocking=True)

        # Pass the input through the neural network to get predictions
        output = net(X)

        # Calculate effective batch size
        B_i = B
        i_end = i_start + B_i

        # Ensure the last batch doesn't exceed the total number of samples
        if i_end > N:
            B_i = N % B
            i_end = i_start + B_i

        # Reshape the output to separate estimated mean and log variance
        output = output.view((B_i, T, 2))

        # Convert log variance to variance (exponentiation)
	# Courtesy of Fredrik K Gustafsson
        output[:, :, 1] = torch.exp(output[:, :, 1])

        # Move the output data to CPU and convert to NumPy array
        out = output.cpu().data[:].numpy()

        # Store the results in the corresponding arrays
        values_out_means[i_start:i_end, :] = out[:B_i, :, 0]
        values_out_vars[i_start:i_end, :] = out[:B_i, :, 1]
        values_gt[i_start:i_end, :] = Y[:B_i, :]

        # Update the starting index for the next batch
        i_start = i_end

    # Return the collected estimated means, estimated variances, and ground truth values
    return (values_out_means, values_out_vars, values_gt)
