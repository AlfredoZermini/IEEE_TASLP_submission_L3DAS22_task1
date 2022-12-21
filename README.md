These scripts were utilized to generate the results in the recent TASLP submission.

# Generate data
To run the scripts, first set the variable in the `paths.env` file.

The data generation requires the data from the L3DAS22 Task 1 challenge (not provided here).

Also, set `config.yaml` for generating the 'train', 'val' and 'test' data. The run the following for each case:

      python prepare_inputs_individual.py

# Train models

To train the models, select either 'MLP' or 'CNN' in the `config.yaml`, then run

      python main.py
      

# Evaluate models

cd into the L3DAS22 folder and run
      
      
# Pre-trained model
https://zenodo.org/record/7427355#.Y5eLyS1Q1QJ

Use 'model_epoch248.hdf5'
