This is a GitHub repository for the biophysics project at the BNEM lab in 2024. \
The project aims to utilize AI model to analyze and predict FRET values based on 5-mer sequences. (T*****TT) 


# env.yaml
   If you want to use these codes, I recommend you to download this file and create this file by executing the code;
   
      conda env create -f env.yaml
   
   Then you can set this environment with this code;
   
      conda activate 2024BNEM


## Directories and Files

# 0. Data preprocess

   Data you need : sequences and FRET values 
   
   - 1_sort.py - Sort sequences alphabetically
   - 2_mean.py - Compute the mean FRET value per sequence
   - 3_pattern.py - Generate all wildcard patterns (e.g. A..CG) and mark match results 
   
# 1. Violin Strip plot

   Data you need 
      Features (X) : Boolean data of patterns with one dot 
                        (e.g. .AAAA: True, .AAAC: False ...) 
      FRET values (Y)

   Generates violin & strip plots, showing FRET variation with a single base changes.
   ![violin_strip_N500_135](https://github.com/user-attachments/assets/2834caa4-4a4c-4d09-a82c-8effee41bb58)
   

# 2. Heatmap
   
      Data you need 
         Features (X) :  Boolean data of patterns with four dots
                           (e.g. A....: True, T....: False ... )
         FRET values (Y)

   Draws heatmaps of average FRET for combined base patterns.
   ![2_N50_1vs1](https://github.com/user-attachments/assets/ef18c627-23d1-4f72-bda1-5b53e53b8851)


# 3. LASSO

      Data you need
         Features (X) : Boolean data of patterns with four dots 
                           (e.g. A....: True, T....: False ... )
         & FRET values (Y)

   Use Lasso regression to predict FRET and outputs feature importances.
   ![image](https://github.com/user-attachments/assets/e887b603-9dbb-4a8c-963b-44613a3b1544)


# 4. Deep Learning

   Predict FRET values using AI models and compare results with a baseline (mean FRET as prediction)

      Data you need : 
         Features (X) : Boolean data of patterns with four dots 
                        (e.g. A....: True, T....: False ... )
         For MLP model, you can use various features such as features such as A.C.., 
         whatever you think is important
         & FRET value (Y) 

   - MLP : Dense(32) > Dense(1), Total params: 705\
   - RNN : SimpleRNN(32) > Dense(1), Total params: 1217\
   - XGBoost (Uses custom MAE objective + Hyperopt to tune hyperparameters)
