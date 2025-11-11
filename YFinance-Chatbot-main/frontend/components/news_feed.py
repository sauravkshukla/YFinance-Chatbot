"""News feed component"""
import streamlit as st
import requests


def render_news_feed(api_url):
    """Render latest market news headlines with expandable full articles"""
    st.markdown("### 📰 Latest Market News")
    st.markdown("*Powered by SerpAPI + Gemini AI*")
    
    with st.spinner("Loading news with AI summaries..."):
        try:
            news_data = requests.get(f"{api_url}/market-news", timeout=60).json()
            
            for idx, article in enumerate(news_data.get("news", [])[:10]):
                # Create unique keys for each article
                article_key = f"news_{idx}"
                fetch_key = f"fetch_{idx}"
                
                # Initialize session state
                if article_key not in st.session_state:
                    st.session_state[article_key] = False
                if fetch_key not in st.session_state:
                    st.session_state[fetch_key] = None
                
                # Article container
                with st.container():
                    col1, col2 = st.columns([6, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div style="padding: 14px; margin: 10px 0; background: linear-gradient(135deg, rgba(31,119,180,0.1) 0%, rgba(255,255,255,0.05) 100%); border-radius: 10px; border-left: 4px solid #1f77b4; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <div style="margin-bottom: 10px;">
                                <strong style="font-size: 17px; color: #fff; line-height: 1.4;">{article['title']}</strong>
                            </div>
                            <div style="color: #bbb; font-size: 14px; line-height: 1.6; margin-bottom: 10px;">
                                {article['summary'][:150]}{'...' if len(article['summary']) > 150 else ''}
                            </div>
                            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                                <div style="display: flex; gap: 15px; flex-wrap: wrap;">
                                    <small style="color: #888;">
                                        📅 {article['published']}
                                    </small>
                                    <small style="color: #888;">
                                        🏢 {article['source']}
                                    </small>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Read More button
                        button_label = "📖 Read" if not st.session_state[article_key] else "🔼 Close"
                        if st.button(button_label, key=f"btn_{article_key}", use_container_width=True):
                            # If opening and not fetched yet, fetch the article
                            if not st.session_state[article_key] and st.session_state[fetch_key] is None:
                                st.session_state[article_key] = True
                                with st.spinner("Fetching full article with AI summary..."):
                                    try:
                                        response = requests.post(
                                            f"{api_url}/fetch-article",
                                            json={"url": article['url']},
                                            timeout=30
                                        )
                                        if response.status_code == 200:
                                            st.session_state[fetch_key] = response.json()
                                        else:
                                            st.session_state[fetch_key] = {"error": "Failed to fetch article"}
                                    except Exception as e:
                                        st.session_state[fetch_key] = {"error": str(e)}
                                st.rerun()
                            else:
                                # Just toggle if already fetched
                                st.session_state[article_key] = not st.session_state[article_key]
                    
                    # Expandable full article
                    if st.session_state[article_key]:
                        article_data = st.session_state[fetch_key]
                        
                        if article_data is None:
                            st.info("⏳ Loading article...")
                        elif article_data.get("success"):
                            content = article_data.get('content', 'No content available')
                            
                            # Display in a nice container
                            st.markdown("""
                            <div style="padding: 16px; margin: 10px 0; background-color: rgba(31,119,180,0.15); border-radius: 8px; border: 1px solid rgba(31,119,180,0.3);">
                                <div style="margin-bottom: 12px;">
                                    <strong style="font-size: 16px; color: #4fc3f7;">📄 Full Article Summary (AI-Generated)</strong>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display content in markdown for better formatting
                            st.markdown(f"""
                            <div style="padding: 16px; margin: 0 0 10px 0; background-color: rgba(31,119,180,0.1); border-radius: 8px;">
                                <div style="color: #ddd; font-size: 14px; line-height: 1.8; white-space: pre-wrap;">
{content}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <a href="{article['url']}" target="_blank" style="display: inline-block; padding: 10px 20px; background-color: #1f77b4; color: white; text-decoration: none; border-radius: 5px; font-size: 14px; font-weight: 500; margin-top: 10px;">
                                🌐 Read Original Article →
                            </a>
                            """, unsafe_allow_html=True)
                        else:
                            st.warning(f"⚠️ Could not fetch full article. {article_data.get('error', '')}")
                            st.markdown(f"""
                            <a href="{article['url']}" target="_blank" style="display: inline-block; padding: 10px 20px; background-color: #1f77b4; color: white; text-decoration: none; border-radius: 5px; font-size: 14px; font-weight: 500;">
                                🌐 Read on Original Site →
                            </a>
                            """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Unable to load news feed. Please check if the backend is running.")
            st.info("💡 Tip: Make sure both SerpAPI and Gemini API keys are configured in backend/.env")
