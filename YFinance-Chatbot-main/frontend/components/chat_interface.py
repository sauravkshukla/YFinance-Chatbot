"""Chat interface component"""
import streamlit as st
import requests
from .charts import (
    create_candlestick_chart, create_line_chart, create_volume_chart, create_dividend_chart,
    create_comparison_chart, create_performance_comparison, create_volume_comparison, 
    create_metrics_comparison, create_scatter_plot, create_correlation_heatmap
)


def render_chat_messages():
    """Display chat message history"""
    # Show welcome message if no messages
    if not st.session_state.messages:
        st.markdown("""
        <div style="padding: 20px; background: linear-gradient(135deg, rgba(31,119,180,0.1) 0%, rgba(255,255,255,0.05) 100%); border-radius: 10px; border-left: 4px solid #1f77b4; margin: 20px 0;">
            <h3 style="color: #1f77b4; margin-top: 0;">👋 Welcome to FinancePilot!</h3>
            <p style="color: #ddd; line-height: 1.6;">
                I'm your AI-powered financial assistant. I can help you with:
            </p>
            <ul style="color: #ddd; line-height: 1.8;">
                <li>📊 <strong>Single Stock Analysis</strong> - "How is AAPL performing?"</li>
                <li>🔄 <strong>Stock Comparisons</strong> - "Compare AAPL vs MSFT"</li>
                <li>📈 <strong>Market Questions</strong> - "What's happening with tech stocks?"</li>
                <li>💡 <strong>Investment Insights</strong> - "Should I invest in NVDA?"</li>
            </ul>
            <p style="color: #4fc3f7; margin-bottom: 0;">
                💬 Just type your question below to get started!
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display chat history
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "chart" in message:
                st.plotly_chart(message["chart"], use_container_width=True)
            
            # Display suggestions if this is the last assistant message
            if message["role"] == "assistant" and "suggestions" in message and message["suggestions"]:
                if idx == len(st.session_state.messages) - 1:  # Only show for last message
                    st.markdown("**💡 You might also want to ask:**")
                    cols = st.columns(min(len(message["suggestions"]), 2))
                    for sug_idx, suggestion in enumerate(message["suggestions"][:4]):
                        with cols[sug_idx % 2]:
                            if st.button(suggestion, key=f"hist_sug_{idx}_{sug_idx}", use_container_width=True):
                                st.session_state.quick_question = suggestion
                                st.rerun()


def handle_chat_input(prompt, ticker, period, api_url):
    """Handle chat input and API response"""
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response from API
    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyzing..."):
            try:
                response = requests.post(
                    f"{api_url}/query",
                    json={
                        "question": prompt,
                        "ticker": ticker,
                        "period": period
                    },
                    timeout=60  # Increased timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result["answer"]
                    data = result["data"]
                    chart_type = result["chart_type"]
                    suggestions = result.get("suggestions", [])
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Create and display chart dynamically
                    chart = None
                    if chart_type == "scatter" and data:
                        chart = create_scatter_plot(data)
                    elif chart_type == "heatmap" and data:
                        chart = create_correlation_heatmap(data)
                    elif chart_type == "comparison" and data:
                        chart = create_comparison_chart(data)
                    elif chart_type == "performance_comparison" and data:
                        chart = create_performance_comparison(data)
                    elif chart_type == "volume_comparison" and data:
                        chart = create_volume_comparison(data)
                    elif chart_type == "metrics_comparison" and data:
                        chart = create_metrics_comparison(data)
                    elif chart_type == "candlestick" and data:
                        chart = create_candlestick_chart(data)
                    elif chart_type == "line" and data:
                        chart = create_line_chart(data)
                    elif chart_type == "volume" and data:
                        chart = create_volume_chart(data)
                    elif chart_type == "bar" and data:
                        chart = create_dividend_chart(data)
                    
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "chart": chart,
                            "suggestions": suggestions
                        })
                    else:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "suggestions": suggestions
                        })
                    
                    # Suggestions will be displayed by render_chat_messages()
                    # No need to display them here
                else:
                    error_msg = f"Error: {response.json().get('detail', 'Unknown error')}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
            
            except Exception as e:
                error_msg = f"Error connecting to API: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
