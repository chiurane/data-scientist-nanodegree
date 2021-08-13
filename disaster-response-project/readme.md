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
## Getting Started
  - Dependencies
  - Installation (locally)
  - Installation (Heroku)
  - Running the Application
    Run the following commands (in this order) in the project's root directory to setup the database and data science assets:
    - To run the ETL pipeline that cleans data and stores the data in the database:
        python -m app.process_data disaster_messages.csv disaster_categories.csv DisasterResponse.db
    - To run the ML pipeline that trains classifier and the pclassifier in pickle file:
        python -m models.train_classifier DisasterResponse.db classifier.pkl
    - To run the webserver and prepare for web requests to the web app:
        python -m app.run
    - Go to http://localhost:3001/ if accessing locally otherwise go to ... on Heroku to access UI.
## Author
[Michael Khumalo](https://github.com/chiurane) ([git](github.com)) or [Michael Khumalo](linkedin.com/chiurane) ([linkedin](linkedin.com))
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
