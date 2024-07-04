This is a GitHub repository for the project at the BNEM lab in 2024. \
The project aims to establish an AI model to predict the FRET value. 

# env.yaml
   If you want to use these codes, I recommend you to download this file and create this file by executing the code;
   
      conda env create -f env.yaml
   
   Then you can set this environment with this code;
   
      conda activate 2024BNEM
   
# 1. Data preprocess
   This folder contains the Python code to sort data and extract features. \
   You can run the code files in the following order (please enter the input and output file names before running) \
    a. sort - Sort the sequences in alphabetical order \
    b. mean - Compute the mean FRET value for each sequence \
    c. pattern - Generate all patterns of 5 bases and determine whether they exist in the sequence. 
   
# 2. AI model
   Develop a Deep learning model to predict FRET values from sequences 

# 3. jul_02
   Plot the violin and strip graph to visualize the difference of FRET when 1 base changes.
