"""FinancePilot - Main Application"""
import streamlit as st
from components.sidebar import render_sidebar
from components.market_overview import render_market_overview
from components.news_feed import render_news_feed
from components.chat_interface import render_chat_messages, handle_chat_input

# Page config
st.set_page_config(
    page_title="FinancePilot",
    page_icon="📈",
    layout="wide"
)

# API endpoint
API_URL = "http://localhost:8000"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "quick_question" not in st.session_state:
    st.session_state.quick_question = None
if "ticker" not in st.session_state:
    st.session_state.ticker = "AAPL"
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "chat"
if "news_loaded" not in st.session_state:
    st.session_state.news_loaded = False

# Title
st.title("📈 FinancePilot")
st.markdown("Your AI-powered financial assistant for stock market analysis and insights")

# Create tabs
tab1, tab2 = st.tabs(["💬 Chat", "📰 News & Market"])

# Chat Tab (Default - Always Active)
with tab1:
    st.markdown("### 💬 Ask Me Anything About Stocks")
    st.markdown("*I can answer questions about any stock, compare stocks, or discuss market trends!*")
    render_chat_messages()

# News & Market Tab (Lazy Load)
with tab2:
    # Only load when tab is clicked
    if st.session_state.get("news_loaded", False):
        render_market_overview(API_URL)
        st.markdown("---")
        render_news_feed(API_URL)
    else:
        # Show loading button
        st.markdown("### 📰 Market News & Overview")
        st.info("👆 Click the button below to load latest market news and overview")
        if st.button("🔄 Load News & Market Data", use_container_width=True, type="primary"):
            st.session_state.news_loaded = True
            st.rerun()

# Sidebar (outside tabs)
ticker, period = render_sidebar()

# Handle quick questions from sidebar and suggestions (outside tabs)
if st.session_state.quick_question:
    prompt = st.session_state.quick_question
    st.session_state.quick_question = None
    # Process the question immediately
    handle_chat_input(prompt, ticker, period, API_URL)
    st.rerun()

# Chat input (must be outside tabs)
prompt = st.chat_input("Ask me anything about stocks... 💬")

if prompt:
    handle_chat_input(prompt, ticker, period, API_URL)
