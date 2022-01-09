# Starbucks Capstone Project (Data Science Nanodegree - Udacity)

## Table of Contents
1. Introduction
2. Getting Started
  - Files in the repo
  - Dependencies
  - Installation (locally)
  - Running the application
5. Author
6. Screenshots
## Introduction
As part of the Data Scientist Nanodegree at Udacity my capstone project is building a recommender system using the Starbucks dataset provided as part of the program. The Starbucks Capstone project is quite exciting not only because its the last project for the program but also because Starbucks is a success story for the ages. The dataset is quite comprehensive with demographic, portfolio (promotions/offers) as well as transactions.
I will analyze the above dataset in detail with the goal of building a recommender system whose goal is to determine which offer or promotion should be given to which customers.

The project is made up of three components:
  1. A [Blog Post](https://medium.com/@chiurane/recommender-system-for-starbucks-7b9f40968e77) describing the project and findings
  2. A process data component which is an ETL for cleaning the Starbucks dataset
  3. A train classifier component which trains the Recommender FunkSVD algorithm and makes predictions given a user
## Getting Started
  - Files in the repository
    |- data - this package contains the ETL components  
    |--- __init__.py - package marker  
    The data is contained in three files:

    |--- portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
    |--- profile.json - demographic data for each customer
    |--- transcript.json - records for transactions, offers received, offers viewed, and offers completed
    Here is the schema and explanation of each variable in the files:

    portfolio.json
    id (string) - offer id
    offer_type (string) - type of offer ie BOGO, discount, informational
    difficulty (int) - minimum required spend to complete an offer
    reward (int) - reward given for completing an offer
    duration (int) - time for offer to be open, in days
    channels (list of strings)
    
    profile.json
    age (int) - age of the customer
    became_member_on (int) - date when customer created an app account
    gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
    id (str) - customer id
    income (float) - customer's income
    
    transcript.json

    event (str) - record description (ie transaction, offer received, offer viewed, etc.)
    person (str) - customer id
    time (int) - time in hours since start of test. The data begins at time t=0
    value - (dict of strings) - either an offer id or transaction amount depending on the record
    
    |--- process_data.py - the script to clean-up data and engineer features  
    |- models - this package contains the machine learning components  
    |--- __init__.py - package marker
    |--- train_classifier.py - this file contains the matrix factorization algorithm implementation
    |- requirements.txt - contains all the libraries required to run the application  
  - Dependencies
      Python 3.7+: The core python language used throughout the project
      NumPy, SciPy, Pandas
      pickle: For persisiting models
  - Installation (locally)
    Simply clone the following data science repo and get cracking:
    git clone https://github.com/chiurane/data-scientist-nanodegree.git
- Running the Application
    Run the following commands (in this order) in the project's root directory to clean the data and build the system:
    - To run the ETL pipeline that cleans data and engineers features:
        python -m data.process_data data/portfolio.json data/profile.json data/transcript.json
    - To run the ML pipeline that trains classifier:
        python -m models.train_classifier train_user_matrix.p test_user_matrix.p

## Author
[Michael Khumalo](https://github.com/chiurane) on [git](https://github.com) or [Michael Khumalo](https://linkedin.com/chiurane) on [linkedin](https://linkedin.com)
## Screenshot
7. Classification Results
    ![Alt text](https://github.com/chiurane/data-scientist-nanodegree/blob/master/disaster-response-project/screenshots/classification_results.PNG, "Classification Results")
8. data.process_data shot sample output
    ![Alt text](https://github.com/chiurane/data-scientist-nanodegree/blob/master/disaster-response-project/screenshots/process_data_shot.PNG "data.process_data sample output")
9. models.train_classifier shot sample output
    ![Alt text](https://github.com/chiurane/data-scientist-nanodegree/blob/master/disaster-response-project/screenshots/models_train_classifier_shot.PNG "models.train_classifier sample output")
