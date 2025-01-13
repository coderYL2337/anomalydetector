# Market Anomaly Detector Bot 📈

A machine learning-powered chatbot that detects market anomalies and provides investment strategy recommendations based on real-time market data.

## Features

- Real-time market data analysis using key indicators:
  - VIX (Volatility Index)
  - DXY (US Dollar Index)
  - 2-Year Treasury Yield (with 4-week MA)
  - 10-Year Treasury Yield (with 4-week MA)
- Machine learning-based anomaly detection
- Custom investment strategy recommendations
- Interactive chat interface
- Detailed market analysis explanations using Groq AI
- Web search capabilities for latest market news (using OpenAI)

## Installation

1. Clone the repository:

git clone https://github.com/coderYL2337/anomalydetector.git
cd anomalydetector

2. Create and activate virtual environment:
   python -m venv anomalydetectorvenv
   source anomalydetectorvenv/bin/activate # On Unix/macOS

   or

  anomalydetectorvenv\Scripts\activate # On Windows

3.Install required packages:

pip install -r requirements.txt

4.Set up environment variables (create a .env file):

OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key

##Usage

1. Run the Streamlit app:

streamlit run app.py

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)
   
3. Interact with the bot:

Ask about market conditions
Request market predictions using "predict market"
Get investment strategy recommendations
Ask about latest market news

##Project Structure

anomalydetector/
├── app.py # Main Streamlit application
├── models/ # selected ML model and scaler
│ ├── logistic_regression_model.pkl
│ └── scaler.pkl
├── othermodels/ # Additional trained models
│ ├── logistic_regression_weighted_model.pkl
│ ├── random_forest_model.pkl
│ └── ...
├── sampledata/ # Training and test datasets
│ └── historical_data.csv
├── anomalyproject.ipynb # Model training & testing notebook
├── requirements.txt # Project dependencies
├── .env # Environment variables (not in git)
├── .gitignore # Git ignore rules
└── README.md # Project documentation

##Model Details
The anomaly detection model is trained on historical market data and uses the following features:

VIX (Volatility Index)
DXY (US Dollar Index)
2-Year Treasury Yield (4-week Moving Average)
10-Year Treasury Yield (4-week Moving Average)

##Dependencies

beautifulsoup4==4.12.3
groq==0.15.0
joblib==1.4.2
numpy==2.2.1
openai==1.59.6
pandas==2.2.3
python-dotenv==1.0.1
requests==2.32.3
scikit-learn==1.6.1
streamlit==1.41.1
yfinance==0.2.51
