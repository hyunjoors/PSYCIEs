# PSYCIEs
2019 SIOP Machine Learning Competition Repository for PSYCIEs

# Instruction for Hyper-Parameter Tuning
1. sub_parameter_dict
This is for CountVectorizer, TfidfTransformer and TruncatedSVD's parameters.  
If you want to change the parameters of these, please edit as you need  

2. parameter_dict
This is for classifiers' parameters.  
When adding a new list of parameters, please follow the format below:  
'name_of_classifier': {  
  'clf__parameter1': <list of values>,  
  'clf__parameter2': <list of values>,  
  'clf__parameter3': <list of values>,  
  ...  
}  
ex.  
'SVR': {  
      'clf__kernel': ['rbf', 'linear'],  
      'clf__C': np.logspace(-2, 6, 9),  
      'clf__gamma': list(np.logspace(-3, 2, 6)),  
      },  

3. To run hyper-parameter tuning, run "python3 param_tuning.py" on your desired IDE


# Instruction for OCEAN Score Prediction
1. OCEAN_model_dict
For each trait, please specify the best estimator.  

2. OCEAN_params_dict
For each trait, please specify the parameters for a corresponding model.
