# FNV4GP
Fast Nested Validation for Genomic Prediction

# Description
This is Python-based, user-friendly software for performing Nested Cross-Validation and Independent Validation to optimize tuning for a wide range of models.

# Key Features 
1) Marker filtering
2) Nested Cross-Validation for a wide range of models
3) Independent Validation
4) Summary of the results   

# Inputs
• Marker filtering
1) Hapmap or Numerical matrix coded 2 (major Homozygous), 0 (minor Homozygous), and 1 (Heterozygous)

• Nested CV
1) Numerical matrix coded 2 (major Homozygous), 0 (minor Homozygous), and 1 (Heterozygous)
2) Adjusted mean trait values such as best linear unbiased estimates (BLUEs)

• Independent Validation
1) Numerical matrix coded 2 (major Homozygous), 0 (minor Homozygous), and 1 (Heterozygous) for the training and testing populations
2) Adjusted mean trait values such as best linear unbiased estimates (BLUEs) for the training and testing populations

• Summary 
1) Outputs from Nested CV modules
2) Outputs from Independent Validation

# Installation
•Install Python 3.12. https://www.python.org/downloads/

•Install the following libraries:
1) tkinter
2) os
3) threading
4) queue
5) pandas
6) numpy
7) sklearn
8) scipy
9) matplotlib

•Download the FNCV4GP GUI app 

•Go to your prompt and run "python FNCV4GP.py" or open it in your Visual Studio and just run the code.
