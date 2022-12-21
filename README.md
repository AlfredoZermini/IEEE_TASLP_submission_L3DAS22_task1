These scripts were utilized to generate the results in the recent TASLP submission.

# Generate data
To run the scripts, first set the variable in the `paths.env` file.

The data generation requires the data from the L3DAS22 Task 1 challenge (not provided here).
https://www.l3das.com/icassp2022/

Also, set `config.yaml` for generating the 'train' and 'val' data. The run the following for each case:

      python prepare_inputs_individual.py

# Train models

To train the models, select either `MLP` or `CNN` in the `config.yaml`, then run

      python main.py
      

# Evaluate models

To evalaute the models, cd into the `L3DAS22` folder. This contains the same scripts provided by the L3DAS challenge, with some modifications to run on the models in this repository. First, set the variables in this the `paths.env` file in the `L3DAS22` folder.

Then, run
      
      python evaluate_tf.py 

The script is configured to run the model available below (to modify it, adjust the saving epoch variable named `best_model_idx`)
      
# Pre-trained model
https://zenodo.org/record/7427355#.Y5eLyS1Q1QJ

Use 'model_epoch248.hdf5'
