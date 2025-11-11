"""Sidebar component"""
import streamlit as st


def render_sidebar():
    """Render the sidebar with settings and quick actions"""
    with st.sidebar:
        st.header("⚙️ Settings")
        st.markdown("*Optional: Select a stock for specific queries*")
        ticker = st.text_input("Stock Ticker", value="AAPL", help="Leave empty for general questions or enter symbol (e.g., AAPL, GOOGL)")
        period = st.selectbox(
            "Time Period",
            ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
            index=2
        )
        
        st.markdown("---")
        st.markdown("### 💡 Try These")
        
        # Quick action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Price Chart", use_container_width=True):
                st.session_state.quick_question = "Show me the price chart"
            if st.button("📈 Performance", use_container_width=True):
                st.session_state.quick_question = "How has this stock performed?"
            if st.button("💰 Dividends", use_container_width=True):
                st.session_state.quick_question = "What are the dividends?"
        
        with col2:
            if st.button("ℹ️ Company Info", use_container_width=True):
                st.session_state.quick_question = "Tell me about this company"
            if st.button("📊 Volume", use_container_width=True):
                st.session_state.quick_question = "Show me the trading volume"
            if st.button("🎯 Analysis", use_container_width=True):
                st.session_state.quick_question = "Give me a detailed analysis"
        
        st.markdown("---")
        st.markdown("### 🔄 Comparisons")
        col3, col4 = st.columns(2)
        with col3:
            if st.button("AAPL vs MSFT", use_container_width=True):
                st.session_state.quick_question = "Compare AAPL vs MSFT"
            if st.button("TSLA vs NVDA", use_container_width=True):
                st.session_state.quick_question = "Compare TSLA vs NVDA"
        with col4:
            if st.button("GOOGL vs META", use_container_width=True):
                st.session_state.quick_question = "Compare GOOGL vs META"
            if st.button("Tech Stocks", use_container_width=True):
                st.session_state.quick_question = "What's happening with tech stocks?"
        
        st.markdown("---")
        st.markdown("### 📝 Example Questions")
        st.markdown("""
        **Single Stock:**
        - What's the P/E ratio?
        - Is this stock overvalued?
        - Should I invest in this?
        
        **Comparisons:**
        - Compare AAPL vs MSFT
        - Which is better: TSLA or NVDA?
        - GOOGL vs META performance
        
        **General:**
        - What's happening in tech stocks?
        - Tell me about the market today
        - Latest news on AI stocks
        """)
        
        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 Popular Stocks")
        popular = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META"]
        selected_stock = st.selectbox("Quick Select", popular, index=0)
        if st.button("Load Stock", use_container_width=True):
            st.session_state.ticker = selected_stock
            st.rerun()
    
    return ticker, period
