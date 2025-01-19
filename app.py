import streamlit as st
import yfinance as yf
import pandas as pd
import joblib
from openai import OpenAI
import os
from dotenv import load_dotenv
from groq import Groq
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Access API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize OpenAI client
openai_client = OpenAI(api_key=openai_api_key)
# Initialize Groq client
groq_client = Groq(api_key=groq_api_key)

# Load model and scaler
model = joblib.load('models/logistic_regression_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Initialize rate limiting in session state
if "last_message_time" not in st.session_state:
    st.session_state.last_message_time = datetime.now()
if "message_count" not in st.session_state:
    st.session_state.message_count = 0

# Add these constants at the top
MESSAGE_LIMIT = 5  # Maximum messages per time window
TIME_WINDOW = 60   # Time window in seconds
MAX_MESSAGES_PER_SESSION = 10  # Maximum total messages per sess


def fetch_data():
    # Regular tickers (VIX and DXY)
    current_tickers = {
        'VIX': '^VIX'      # Volatility Index
    }
    

    
    # Dictionary to store all data
    data = {}

    # Add timestamp
    data['timestamp'] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S EST")
    
    
    # Fetch current prices for VIX and DXY
    for name, ticker in current_tickers.items():
        try:
            ticker_data = yf.Ticker(ticker)
            history = ticker_data.history(period="1d")
            if not history.empty and history['Close'].iloc[-1] is not None:
                data[name] = history['Close'].iloc[-1]
                print(f"Successfully fetched {name} from yfinance: {data[name]}")
            else:
                print(f"Falling back to web scraping for {name}")
                # Fallback to web scraping if yfinance fails
                url = f"https://finance.yahoo.com/quote/{ticker}"
                try:
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    response = requests.get(url, headers=headers)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    price = soup.find('fin-streamer', {'data-field': 'regularMarketPrice'})
                    if price:
                        price_text = price.text.strip().replace(',', '')
                        data[name] = float(price_text)
                        print(f"Successfully scraped {name}: {data[name]}")
                    else:
                        print(f"Could not find price data for {name} in scraped content")
                        data[name] = None
                except Exception as e:
                    print(f"Web scraping failed for {name}: {str(e)}")
                    data[name] = None
        except Exception as e:
            print(f"Error fetching {name}: {str(e)}")
            data[name] = None
    
    return data

# Predict anomalies using the model
def predict_anomalies(data):
    input_data = {
        'VIX': data['VIX']
    }
    df = pd.DataFrame([input_data])
    scaled_data = scaler.transform(df)
    probability = model.predict_proba(scaled_data)[:, 1][0]
    label = 1 if probability >= 0.7 else 0
    return label, probability

# Recommend strategy
def recommend_strategy(probability):
    if probability > 0.7:
        return "Move to safer investments (bonds, gold, cash)"
    elif probability < 0.3:
        return "Consider balanced portfolio with both equities and safer assets"
    else:
        return "Maintain cautious approach with diversified portfolio"

# Use Llama (via GROQ API) for detailed explanations
def explain_strategy(strategy, market_data,probability):
    context = f"""Current market conditions:
    VIX: {market_data['VIX']:.2f}"""  
    
    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "assistant",
                "content": "You are a financial expert explaining investment strategies based on market conditions."
            },
            {
                "role": "user",
                "content": f"""Given these market conditions:\n{context}\n
While the quantitative model suggests {probability*100:.1f}% probability of market anomaly,
please analyze the current market conditions and explain whether you agree or disagree with the recommendation to '{strategy}',
providing detailed reasoning for your conclusion."""
            }
        ],
        temperature=0.7,
        max_tokens=1024,
        top_p=1
    )
    response = completion.choices[0].message.content
    print(f"[Groq Response: {response}")
    return completion.choices[0].message.content


# Modify the chat_with_openai function to handle web searches specifically
def chat_with_openai(conversation):
    # Try different models in order
    models = ["gpt-4o-mini", "gpt-4o", "gpt-4"]
    
    for model in models:
        try:
            completion = openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "assistant", "content": "You are a financial expert ready to fetch data online if necessary to help answer users questions."},
                    *conversation
                ]
            )
            response = completion.choices[0].message.content
            print(f"[OpenAI Response: {response}")
            
            return completion.choices[0].message.content
        except Exception as e:
            if model == models[-1]:  # If this was the last model to try
                return "I apologize, but I'm currently unable to search the web for this information. Would you like me to analyze the market data I have available instead?"
            continue


# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hello! I'm your Market Anomaly Detector Bot. I can help you analyze market conditions and detect potential anomalies using real-time data and machine learning. Feel free to ask me about market conditions or type 'predict market' for a detailed analysis."}
    ]

# Streamlit UI
st.title("Market Anomaly Detector Bot")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input using st.chat_input instead of st.text_input
if prompt := st.chat_input("What would you like to know about the market?"):
    # Check rate limits
    current_time = datetime.now()
    time_diff = (current_time - st.session_state.last_message_time).total_seconds()
    
    # Reset counter if time window has passed
    if time_diff > TIME_WINDOW:
        st.session_state.message_count = 0
        st.session_state.last_message_time = current_time
    
    # Check rate limits
    if st.session_state.message_count >= MESSAGE_LIMIT:
        st.error(f"Rate limit exceeded. Please wait {TIME_WINDOW - time_diff:.0f} seconds.")
    elif len(st.session_state.messages) >= MAX_MESSAGES_PER_SESSION:
        st.error("Maximum session message limit reached. Please refresh the page to start a new session.")
    else:
        # If we get here, proceed with the message
        st.session_state.message_count += 1
        st.session_state.last_message_time = current_time
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
    
        # Store market data in session state if not already present
        if 'market_data' not in st.session_state:
            with st.spinner('Fetching market data...'):
                st.session_state.market_data = fetch_data()
        
        # Check for specific market status questions first
        if any(phrase in prompt.lower() for phrase in ["market open", "market closed", "is market open", "is market closed"]):
            # Get current time in ET
            et_time = pd.Timestamp.now(tz='US/Eastern')
            is_weekend = et_time.weekday() >= 5
            is_market_hours = 9 <= et_time.hour < 16
            
            if is_weekend:
                response = "The US stock market is currently closed (weekend)."
            elif is_market_hours:
                response = f"The US stock market is currently open. Current time (ET): {et_time.strftime('%I:%M %p ET')}"
            else:
                response = f"The US stock market is currently closed. Regular trading hours are 9:30 AM - 4:00 PM ET. Current time: {et_time.strftime('%I:%M %p ET')}"

        # If the query suggests needing web search
        if any(keyword in prompt.lower() for keyword in ["news", "latest", "current events", "today",'current','now','recent']):
            response = chat_with_openai(st.session_state.messages)
        # For market analysis and predictions
        elif any(keyword in prompt.lower() for keyword in ["market", "crash", "predict", "risk"]):
            with st.spinner('Fetching market data...'):
                market_data = fetch_data()
            
            label, prob = predict_anomalies(st.session_state.market_data)
            strategy = recommend_strategy(prob)
            explanation = explain_strategy(strategy, st.session_state.market_data,prob)  # Pass market_data to explain_strategy

            # Only include market data if it hasn't been shown before
            if len(st.session_state.messages) <= 2:  # First real interaction
                market_info = f"""Current market indicators (as of {st.session_state.market_data['timestamp']}):

    VIX (Volatility Index): {st.session_state.market_data['VIX']:.2f}"""
            else:
                market_info = ""
            
            response = f"""{market_info}
    data_timestamp: {st.session_state.market_data['timestamp']}
    
    Market Analysis:
    
    Binary Anomaly Value: {label} (0 = Normal, 1 = Anomaly)
    The model indicates a {prob*100:.1f}% probability of market anomaly.
    Risk Level: {"High" if label == 1 else "Normal"}

    Recommended Strategy: {strategy}

    Detailed Analysis:
    {explanation}"""
        # For general questions, use Groq
        else:
            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "assistant",
                        "content": f"""You are a financial expert providing guidance based on current market conditions as of {st.session_state.market_data['timestamp']}:
                        VIX: {st.session_state.market_data['VIX']:.2f}"""
                    },
                    *st.session_state.messages[-2:]  # Only pass the last interaction for context
                ],
                temperature=0.7,
                max_tokens=1024
            )
            response = completion.choices[0].message.content

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)

# Add refresh button for market data
if st.button("Refresh Market Data"):
    with st.spinner('Refreshing market data...'):
        st.session_state.market_data = fetch_data()
        st.success(f"Market data updated as of {st.session_state.market_data['timestamp']}")

# Add some styling
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    div.stMarkdown {
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)