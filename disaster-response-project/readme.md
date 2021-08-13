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
Michael Khumalo (git) or Michael Khumalo (linkedin)
## Screenshot
1. Top Navigation Home
    ![Alt text](/screenshots/top_nav_home.png?raw=true "Top Navigation")
3. Overview of Training Dataset
4. Classification Results
5. app.run shot sample output
6. data.process_data shot sample output
7. models.train_classifier shot sample output
