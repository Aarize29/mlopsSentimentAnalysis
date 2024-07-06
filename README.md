# Sentiment Analysis Project

This project is a sentiment analysis application that classifies tweets as positive or negative. The application is built using Python and Flask and containerized using Docker. The dataset consists of labeled tweets, and the best-performing model is selected from multiple machine learning models.

## Project Structure

sentiment_analysis/
│
├── app.py # Flask application
├── Dockerfile # Docker configuration file
├── labeled_tweets.csv # CSV file containing the labeled tweet data
├── requirements.txt # Python dependencies
├── sentiment_analysis.py # Data preprocessing, training models, and saving the best model
├── .gitignore # Git ignore file
├── README.md # Project documentation
├── venv/ # Virtual environment (not included in version control)
└── .github/ # GitHub configuration directory
└── workflows/ # GitHub Actions workflows directory
└── ci-cd.yml # CI/CD pipeline configuration file


## Getting Started

### Prerequisites

- Python 3.x
- Docker

### Set Up Virtual Environment and Install Dependencies

1. **Create Virtual Environment**

    ```bash
    cd sentiment_analysis
    python -m venv venv
    ```

2. **Activate Virtual Environment**

    - On Windows:

        ```bash
        .\venv\Scripts\Activate.ps1
        ```

    - On macOS and Linux:

        ```bash
        source venv/bin/activate
        ```

3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

### Preprocess Data and Train Models

Run the `sentiment_analysis.py` script to preprocess the data, train the models, and save the best model:

```bash
python sentiment_analysis.py
```
### Test Flask App Locally
Run the Flask app locally to test the sentiment prediction API:

```bash
python app.py
```
Send a POST request to http://127.0.0.1:5000/predict with a JSON body containing the text to predict. For example, using PowerShell:
```bash
$headers = @{'Content-Type' = 'application/json'}
$body = '{"text": "I love this product!"}'
$response = Invoke-WebRequest -Uri http://127.0.0.1:5000/predict -Method POST -Headers $headers -Body $body
$response.Content
```
### Using the Docker Image
To run the Docker image on any machine with Docker installed:
```bash
docker pull aarize/sentiment-analysis:latest
docker run --rm -p 3000:5000 your_dockerhub_username/sentiment-analysis:latest
```



