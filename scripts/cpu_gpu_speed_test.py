"""
Simple speed test for CPU and GPU using pytorh.
"""

# Load packages.
import time
import torch


# CPU.
start_time = time.time()
a = torch.ones(4000, 4000)
for _ in range(1000):
    a += a
elapsed_time = time.time() - start_time
print(f">>> CPU time = {elapsed_time}")


# GPU.
start_time = time.time()
b = torch.ones(4000, 4000).cuda()
for _ in range(1000):
    b += b
elapsed_time = time.time() - start_time
print(">>> GPU time = {elapsed_time}")
