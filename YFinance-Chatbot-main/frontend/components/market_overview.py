"""Market overview component"""
import streamlit as st
import requests


def render_market_overview(api_url):
    """Render market overview with top gainers, losers, and most active"""
    st.markdown("### 📊 Market Overview")
    
    with st.spinner("Loading market data..."):
        try:
            market_data = requests.get(f"{api_url}/market-overview", timeout=10).json()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 🚀 Top Gainers")
                for stock in market_data.get("gainers", [])[:5]:
                    change_color = "#00ff00"
                    st.markdown(f"""
                    <div style="padding: 10px; margin: 6px 0; background-color: rgba(0,255,0,0.05); border-left: 3px solid #00ff00; border-radius: 5px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong style="color: #1f77b4; font-size: 16px;">{stock['ticker']}</strong><br>
                                <small style="color: #aaa;">{stock['name'][:25]}</small>
                            </div>
                            <div style="text-align: right;">
                                <strong style="font-size: 16px;">${stock['price']:.2f}</strong><br>
                                <small style="color: {change_color}; font-weight: bold;">+${abs(stock['change']):.2f} (+{stock['change_pct']:.2f}%)</small>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### 📉 Top Losers")
                for stock in market_data.get("losers", [])[:5]:
                    change_color = "#ff4444"
                    st.markdown(f"""
                    <div style="padding: 10px; margin: 6px 0; background-color: rgba(255,68,68,0.05); border-left: 3px solid #ff4444; border-radius: 5px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong style="color: #1f77b4; font-size: 16px;">{stock['ticker']}</strong><br>
                                <small style="color: #aaa;">{stock['name'][:25]}</small>
                            </div>
                            <div style="text-align: right;">
                                <strong style="font-size: 16px;">${stock['price']:.2f}</strong><br>
                                <small style="color: {change_color}; font-weight: bold;">${stock['change']:.2f} ({stock['change_pct']:.2f}%)</small>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("#### 🔥 Most Active")
                for stock in market_data.get("active", [])[:5]:
                    change_color = "#00ff00" if stock["change_pct"] > 0 else "#ff4444"
                    sign = "+" if stock["change_pct"] > 0 else ""
                    st.markdown(f"""
                    <div style="padding: 10px; margin: 6px 0; background-color: rgba(255,255,255,0.05); border-left: 3px solid #ffa500; border-radius: 5px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong style="color: #1f77b4; font-size: 16px;">{stock['ticker']}</strong><br>
                                <small style="color: #aaa;">{stock['name'][:25]}</small>
                            </div>
                            <div style="text-align: right;">
                                <strong style="font-size: 16px;">${stock['price']:.2f}</strong><br>
                                <small style="color: {change_color}; font-weight: bold;">{sign}${abs(stock['change']):.2f} ({sign}{stock['change_pct']:.2f}%)</small>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        except Exception as e:
            st.error("Unable to load market data. Please check if the backend is running.")
