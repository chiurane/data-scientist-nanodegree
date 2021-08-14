# Disaster Response Pipeline Project (Data Science Nanodegree - Udacity)

## Table of Contents
1. Introduction
2. Getting Started
  - Dependencies
  - Installation (locally)
  - Installation (Heroku)
  - Running the application
5. Author
6. Screenshots
## Introduction
There are a number of use cases for classifying text into categories. Classifying messages to enable efficiency and effective deployment during a disaster is one of those use cases. We focus on this in the Disaster Response Pipeline Project. Using data from Figure 8 we train a classifier to automacally classifier messages and build a front so that users can view classifications in realtime.
The project is made up of three components:
  1. A web application front where messages can be entered and classification results displayed to the user
  2. A data component which is an ETL for cleaning the Figure 8 data and storing it in a sqlite database
  3. A models component which trains a classifier on the data set and stores the results a pickle file that call be queried in realtime to show classification results.
## Getting Started
  - Dependencies
      Python 3.7+: The core python language used throughout the project
      NumPy, SciPy, Pandas, Sciki-Learn: Machine Learning pipeline libraries
      NLTK: A Natural Langauge Processing (NLP) for NLP functions
      sqlalchemy: sqlite database for storing processed messages from raw csv files. 
      pickle: For persisiting models
      Flask, Plotly: Presentation layer for our web application and visualizations
      Heroku: platform for deploying and running application workloads
  - Installation (locally)
    Simply clone the following data science repo and get cracking:
    git clone https://github.com/chiurane/data-scientist-nanodegree.git
    The file classifier.pkl is to big to updload here so you will have to run "python -m models.train_classifier" to get the pkl file on your own machine
  - Running the Web Apop (Heroku)
    The Web has been deployed to Heroku and can be accessed by running [DistasterResponsePipeline](https://dsnd-drs-proj.herokuapp.com/)
  - Running the Application
    Run the following commands (in this order) in the project's root directory to setup the database and data science assets:
    - To run the ETL pipeline that cleans data and stores the data in the database:
        python -m app.process_data disaster_messages.csv disaster_categories.csv DisasterResponse.db
    - To run the ML pipeline that trains classifier and the pclassifier in pickle file:
        python -m models.train_classifier DisasterResponse.db classifier.pkl (Train classifier not uploaded to git because its too big.
    - To run the webserver and prepare for web requests to the web app:
        python -m app.run
    - Go to http://localhost:3001/ if accessing locally otherwise go to ... on Heroku to access UI.
## Author
[Michael Khumalo](https://github.com/chiurane) on [git](https://github.com) or [Michael Khumalo](https://linkedin.com/chiurane) on [linkedin](https://linkedin.com)
## Screenshot
1. Top Navigation Home
    ![Alt text](https://github.com/chiurane/data-scientist-nanodegree/blob/master/disaster-response-project/screenshots/top_nav_home.PNG "Top Navigation")
3. Overview of Training Dataset
    ![Alt text](https://github.com/chiurane/data-scientist-nanodegree/blob/master/disaster-response-project/screenshots/overview_of_training_dataset.PNG "Overview of Training Dataset")
5. Classification Results
    ![Alt text](https://github.com/chiurane/data-scientist-nanodegree/blob/master/disaster-response-project/screenshots/classification_results.PNG, "Classification Results")
7. app.run shot sample output
  ![Alt text](https://github.com/chiurane/data-scientist-nanodegree/blob/master/disaster-response-project/screenshots/app_run_shot.PNG "app.run sample output")
9. data.process_data shot sample output
    ![Alt text](https://github.com/chiurane/data-scientist-nanodegree/blob/master/disaster-response-project/screenshots/process_data_shot.PNG "data.process_data sample output")
11. models.train_classifier shot sample output
    ![Alt text](https://github.com/chiurane/data-scientist-nanodegree/blob/master/disaster-response-project/screenshots/models_train_classifier_shot.PNG "models.train_classifier sample output")
