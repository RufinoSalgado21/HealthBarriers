# Multilabel and Multiclass Analysis of Social Determinants of Health
##Project Overview
This project involved preprocessing datasets of patient demographic information (primary languages, income levels, etc.) and medical history (visit information, case management notes, etc.). The resulting dataset was used to train multilabel and multiclass classification models. Their predictions and correlational relationships between features were used to determine the most influential factors in predicting barriers to healthcare, actions providers would need to take in reponse, and the intensity of visits for patients. These results were also used to provide suggestions for future work on building predictive models for healthcare providers.

##Preprocessing

Datasets were provided by Northwestern researchers and healthcare providers. Four datasets were provided: two describing patients from DuPage county and two describing patients from the Chinatown neighborhood in Chicago. Each pair of datasets including one dedicated to patient demographic information and the other for medical histories.

Datasets for each location had to be merged according to patient ID's to ensure each patient would have a complete set of data to analyze. In total, 674 complete sets of patient data. 

In order to merge all datasets into one consolidated set, data had to be reformated for each feature to have consistent values. Lists of three most frequent barriers encountered by each patient were created as validation data for model training. The same was done for actions taken by healthcare providers. The number of individual visits recorded per patient 
