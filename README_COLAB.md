# Running DOTA-Mono Training on Google Colab

This guide explains how to run the DOTA-Mono training pipeline on Google Colab.

## Prerequisites

1. A Google account
2. Access to Google Colab (https://colab.research.google.com/)
3. Your data files (`train.h5` and `test.h5`) uploaded to Google Drive

## Setup Instructions

1. **Upload Data Files**
   - Create a folder named `DOTA-Mono` in your Google Drive
   - Inside it, create the following structure:
     ```
     DOTA-Mono/
     ├── data/
     │   ├── training/
     │   │   └── train.h5
     │   └── test/
     │       └── test.h5
     ```

2. **Open the Colab Notebook**
   - Go to https://colab.research.google.com/
   - Click on "File" > "Upload notebook"
   - Upload the `mono_training_colab.ipynb` file

3. **Run the Notebook**
   - The notebook will:
     - Install required packages
     - Mount your Google Drive
     - Clone the DOTA-Mono repository
     - Copy your data files
     - Run the mono-energetic data extraction
     - Train the model
     - Save results back to your Google Drive

4. **Monitor Training**
   - The training progress will be displayed in the notebook
   - Results will be saved to:
     - `DOTA-Mono/mono_energy_validation/` - Contains extracted data and validation results
     - `DOTA-Mono/weights/` - Contains trained model weights

## Notes

- Make sure to use a GPU runtime in Colab (Runtime > Change runtime type > GPU)
- The training process may take several hours depending on your data size
- All results will be automatically saved to your Google Drive

## Troubleshooting

If you encounter any issues:
1. Check that your data files are correctly uploaded to Google Drive
2. Ensure you're using a GPU runtime
3. Verify that all required packages are installed
4. Check the Colab logs for any error messages 