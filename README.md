This is a GitHub repository for the biophysics project at the BNEM lab in 2024. \
The project aims to utilize AI model to analyze and predict FRET values based on 5-mer sequences. (T*****TT) 


## env.yaml
   If you want to use these codes, I recommend you to download this file and create this file by executing the code;
   
      conda env create -f env.yaml
   
   Then you can set this environment with this code;
   
      conda activate 2024BNEM


# Directories and Files

## 0. Data preprocess

   Required data: DNA sequences and their corresponding FRET values 
   
   - 1_sort.py - Sort sequences alphabetically
   - 2_mean.py - Compute the mean FRET value per sequence
   - 3_pattern.py - Generate all wildcard patterns (e.g. A..CG) and label whether each sequence matches the patterns 
   
## 1. Violin Strip plot

      Required data
         Features (X) :  Boolean data of patterns with one wildcard
                           (e.g. AAAA.: True, AAAT.: False ... )
         FRET values (Y)

   Generates violin and strip plots to show how FRET values vary with single-base substitutions.
   Plots below indicate that sequences with 'A' at the second position tend to exhibit slightly higher FRET values (see leftmost blue-dotted plots in each grid)
   
   ![violin_strip_N500_135](https://github.com/user-attachments/assets/2834caa4-4a4c-4d09-a82c-8effee41bb58)
   

## 2. Heatmap
   
      Data you need 
         Features (X) :  Boolean data of patterns with four wildcards
                           (e.g. A....: True, T....: False ... )
         FRET values (Y)

   Draws heatmaps of average FRET for combined base patterns.
   Sequences with first T (T....) or second A (.A...) tend to appear in paler blue, indicating higher FRET values.
   
   ![2_N50_1vs1](https://github.com/user-attachments/assets/ef18c627-23d1-4f72-bda1-5b53e53b8851)


## 3. LASSO

      Required data
         Features (X) : Boolean data of patterns with four wildcards 
                           (e.g. A....: True, T....: False ... )
         & FRET values (Y)

   Applies Lasso regression to predict FRET values and extracts feature importances, identifying which bases contribute most under a linear assumption.
   
   ![image](https://github.com/user-attachments/assets/e887b603-9dbb-4a8c-963b-44613a3b1544)


## 4. Deep Learning

   Predict FRET values using machine learning models and compare them to a baseline (mean FRET as prediction)

      Requried data : 
         Features (X) : Boolean data of patterns with four wildcards 
                        (e.g. A....: True, T....: False ... )
         The MLP model can utilize various features such as A.C.., whatever you think is important.
         & FRET value (Y) 

   - MLP : Dense(32) -> Dense(1), Total params: 705
   - RNN : SimpleRNN(32) -> Dense(1), Total params: 1217
   - XGBoost (Uses custom MAE objective + Hyperopt to tune hyperparameters)

     All AI models outperform the baseline.
     Moreover, deep learning models(MLP, MLP_pair, RNN) shows better performance than the tree-based XGBoost model.
      
     ![Analysis and Prediction of FRET](https://github.com/user-attachments/assets/8817ac92-43db-4574-b0f3-765c939cbe1c)

