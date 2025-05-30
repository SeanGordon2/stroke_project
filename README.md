# Stroke Prediction Project

This project utilises a healthcare dataset related to cerebrovascular events to build predictive tools. The dataset is 
sourced from Kaggle and contains patient information such as gender; age; comorbidities such as hypertension and heart 
disease; marital status; type of work; residence; average glucose level; BMI; smoking status; and previous history of 
a stroke.

## Dataset

**Source**: [Kaggle - Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

**File**: 'healthcare-dataset-stroke-data.csv'

### Download Instructions (via Kaggle API)

1. **Install Kaggle CLI**:  
pip install kaggle
2. **Set up Kaggle API key**  
Create New API Token on Kaggle  
mkdir -p ~/.kaggle  
mv /path/to/kaggle.json ~/.kaggle/  
chmod 600 ~/.kaggle/kaggle.json  
3. **Download the Dataset**  
kaggle datasets download -d fedesoriano/stroke-prediction-dataset  
unzip stroke-prediction-dataset.zip  

