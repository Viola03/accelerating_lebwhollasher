#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Check if the correct number of arguments is provided
if len(sys.argv) != 2:
    print("Usage: ./visualtest.py <filename>")
    sys.exit(1)

file = sys.argv[1] #csv from command line

csv_file_path = rf"outputs\{file}" 
df = pd.read_csv(csv_file_path, skiprows=9,  sep='\s+')
df.columns = ['MC_Step', 'Ratio', 'Energy', 'Order']

plt.figure(figsize=(12, 5))

# Subplot for Energy vs MC_Step
plt.subplot(1, 2, 1)
plt.plot(df['MC_Step'], df['Energy'], marker='o', color='r')
plt.title('T = 0.5')
plt.xlabel('MCS')
plt.ylabel('Reduced Energy')
plt.grid()

# Subplot for Order vs MC_Step
plt.subplot(1, 2, 2)
plt.plot(df['MC_Step'], df['Order'], marker='o', color='b')
plt.title('T = 0.5')
plt.xlabel('MCS')
plt.ylabel('Order')
plt.grid()

plt.tight_layout()
plt.show()