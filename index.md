# Multilabel and Multiclass Analysis of Social Determinants of Health
## Project Overview

This project involved preprocessing datasets of patient demographic information (primary languages, income levels, etc.) and medical history (visit information, case management notes, etc.). The resulting dataset was used to train multilabel and multiclass classification models. Their predictions and correlational relationships between features were used to determine the most influential factors in predicting **barriers to healthcare**, **actions taken in response** by providers, and the **intensity of visits** for patients. These results were also used to provide suggestions for future work on building predictive models for healthcare providers.

## Preprocessing

Datasets were provided by Northwestern researchers and healthcare providers. Four datasets were provided: two describing patients from DuPage county and two describing patients from the Chinatown neighborhood in Chicago. Each pair of datasets including one dedicated to patient demographic information and the other for medical histories.

Datasets for each location had to be merged according to patient ID's to ensure each patient would have a complete set of data to analyze. In total, **674** complete sets of patient data. 

In order to merge all datasets into one consolidated set, data had to be reformated for each feature to have consistent values. To extract further features from existing data, the Google Maps API was used with patients' zip code information to determine nearest hospitals, distance to those hospitals, and predicted travel times. Lists of three most frequent barriers encountered by each patient were created as validation data for model training. The same was done for actions taken by healthcare providers. The number of individual visits recorded per patient were also counted as validation data for multiclass model training.

The merged dataset consisted of **12 features**: 
- age 
- primary language
- birth country
- marital status 
- education level
- household size
- income level
- employment status
- visit count 
- distance to a hospital in km
- predicted driving time to hospital
- predicted public transit time to hospital

## Modeling
A multilabel classification model was trained using the dataset to predict the three most likely barriers to healthcare patients and another was trained to predict the three most likely action needed to be taken. The model was set up using **Python** and the **Keras API**. The multilabel models used were neural networks consisting of Convolutional layers, MaxPooling layers, Dense layers, and Dropout layers. The multiclass classification of visit intensity was done with a Support Vector Machine algorithm, specifically the LinearSVC function. Average performance metrics (accuracy, precision, recall, and F1 scores) were calculated over the course of a K-fold cross validation. These average metrics were determined for individual barrier and action classes as well as for all classes as a whole.

## Analysis

Correlational relationships between features and the barriers reported by patients suggest primary languages



