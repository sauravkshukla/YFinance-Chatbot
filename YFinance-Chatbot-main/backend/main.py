from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests
import os
from typing import Optional
import google.generativeai as genai
from dotenv import load_dotenv
from opensearch_client import get_vector_db
import hashlib

load_dotenv()

app = FastAPI(title="FinancePilot API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# NocoDB configuration
NOCODB_URL = os.getenv("NOCODB_URL", "http://localhost:8080")
NOCODB_TOKEN = os.getenv("NOCODB_TOKEN", "")
NOCODB_TABLE_ID = os.getenv("NOCODB_TABLE_ID", "")

# Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        print("✅ Gemini 2.0 Flash configured successfully")
    except Exception as e:
        print(f"⚠️ Gemini 2.0 Flash configuration error: {e}")
        model = None
else:
    model = None
    print("⚠️ No Gemini API key found, using fallback mode")

# SerpAPI configuration
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")

# Initialize OpenSearch Vector DB (disabled for performance)
# Uncomment when OpenSearch is running
# vector_db = get_vector_db()
# if vector_db and vector_db.client:
#     vector_db.create_index("news_articles")
#     vector_db.create_index("stock_data")
#     vector_db.create_index("chat_history")
vector_db = None  # Disabled for now

# Simple in-memory cache for news (5 minutes TTL)
news_cache = {"data": None, "timestamp": None}
CACHE_TTL = 300  # 5 minutes

class QueryRequest(BaseModel):
    question: str
    ticker: Optional[str] = None
    period: Optional[str] = "1mo"
    compare_tickers: Optional[list] = None  # For stock comparisons

class QueryResponse(BaseModel):
    answer: str
    data: dict
    chart_type: str
    suggestions: list = []

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    needs_ticker: bool = False
    suggestions: list = []

def save_to_nocodb(question: str, ticker: str, response: dict):
    """Save query history to NocoDB"""
    if not NOCODB_TOKEN or not NOCODB_TABLE_ID:
        return
    
    try:
        headers = {
            "xc-token": NOCODB_TOKEN,
            "Content-Type": "application/json"
        }
        data = {
            "question": question,
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "response": str(response)
        }
        requests.post(
            f"{NOCODB_URL}/api/v1/db/data/noco/{NOCODB_TABLE_ID}",
            headers=headers,
            json=data,
            timeout=5
        )
    except Exception as e:
        # Silently fail if NocoDB is not available
        pass

def get_stock_data(ticker: str, period: str):
    """Fetch comprehensive stock data"""
    try:
        # Create ticker with proper headers to avoid rate limiting
        stock = yf.Ticker(ticker)
        
        # Fetch history - yfinance v0.2.32+ uses different method
        hist = pd.DataFrame()
        try:
            hist = stock.history(period=period, interval="1d", actions=False, auto_adjust=True, back_adjust=False, repair=True, keepna=False, proxy=None, rounding=False, timeout=30)
        except Exception as e:
            print(f"History fetch error: {e}")
            # Try alternative method
            try:
                import yfinance.shared as shared
                shared._ERRORS.clear()
                hist = stock.history(period=period)
            except:
                pass
        
        # Get info with fallback
        info = {}
        try:
            info = stock.info
            if not info or len(info) == 0:
                # Try fast_info as fallback
                try:
                    fast_info = stock.fast_info
                    info = {
                        "longName": ticker,
                        "currentPrice": fast_info.get("lastPrice", 0),
                        "marketCap": fast_info.get("marketCap", 0),
                        "sector": "N/A"
                    }
                except:
                    info = {"longName": ticker, "currentPrice": 0, "sector": "N/A"}
        except Exception as e:
            print(f"Info fetch error: {e}")
            info = {"longName": ticker, "currentPrice": 0, "sector": "N/A"}
        
        # Get dividends
        dividends = pd.Series()
        try:
            dividends = stock.dividends
        except Exception as e:
            print(f"Dividends fetch error: {e}")
        
        # If we have history, update current price from it
        if not hist.empty and "currentPrice" not in info:
            info["currentPrice"] = float(hist["Close"].iloc[-1])
        
        return {
            "history": hist,
            "info": info,
            "dividends": dividends,
            "ticker": ticker
        }
    except Exception as e:
        print(f"Stock data error: {e}")
        raise HTTPException(status_code=400, detail=f"Unable to fetch data for {ticker}. Please check the ticker symbol and try again.")

def analyze_with_gemini(question: str, stock_data: dict):
    """Use Gemini with RAG to understand user intent and generate response"""
    if not model:
        # Fallback to simple keyword matching
        return analyze_without_gemini(question, stock_data)
    
    # RAG: Retrieve relevant context from OpenSearch (only if available and connected)
    relevant_context = ""
    # Skip RAG if OpenSearch is not properly connected
    # This prevents timeouts when OpenSearch is unavailable
    
    try:
        ticker = stock_data["ticker"]
        info = stock_data["info"]
        hist = stock_data["history"]
        dividends = stock_data["dividends"]
        
        # Calculate additional metrics
        price_change = 0
        price_change_pct = 0
        if not hist.empty and len(hist) > 1:
            price_change = hist["Close"].iloc[-1] - hist["Close"].iloc[0]
            price_change_pct = (price_change / hist["Close"].iloc[0]) * 100
        
        # Prepare comprehensive context for Gemini with RAG
        context = f"""
You are an intelligent financial assistant with access to real-time stock data and recent news. Be conversational, helpful, and insightful.

Current Stock: {ticker}
Company: {info.get('longName', ticker)}
Sector: {info.get('sector', 'N/A')}
Industry: {info.get('industry', 'N/A')}

Current Metrics:
- Current Price: ${info.get('currentPrice', 0):.2f}
- Market Cap: ${info.get('marketCap', 0):,.0f}
- 52 Week High: ${info.get('fiftyTwoWeekHigh', 0):.2f}
- 52 Week Low: ${info.get('fiftyTwoWeekLow', 0):.2f}
- P/E Ratio: {info.get('trailingPE', 'N/A')}
- EPS: ${info.get('trailingEps', 0):.2f}
- Dividend Yield: {info.get('dividendYield', 0) * 100:.2f}%
- Beta: {info.get('beta', 'N/A')}
- Volume: {info.get('volume', 0):,}
- Average Volume: {info.get('averageVolume', 0):,}

Recent Performance:
- Price Change: ${price_change:.2f} ({price_change_pct:+.2f}%)
- Historical Data Points: {len(hist)} days
- Has Dividends: {'Yes' if not dividends.empty else 'No'}
{relevant_context}
User Question: "{question}"

Instructions:
1. Provide a clear, well-structured answer using proper formatting
2. Use bullet points (•) for lists and key metrics
3. Use line breaks for better readability
4. Highlight important numbers and percentages
5. Be conversational but professional
6. Structure your response with clear sections when appropriate

Format Guidelines:
- Start with a brief overview (1-2 sentences)
- Use bullet points for key metrics
- Add blank lines between sections
- Use clear, concise language
- Avoid run-on sentences

Determine the best chart to show based on the user's request:
- "scatter" for scatter plots, correlation analysis, relationship between stocks
- "heatmap" for correlation matrix, multiple stock correlations
- "comparison" for comparing stock prices over time
- "performance_comparison" for normalized % change comparison
- "volume_comparison" for comparing trading volumes
- "metrics_comparison" for comparing P/E, market cap, dividends
- "candlestick" for price action, OHLC data, trading patterns
- "line" for simple price trends over time
- "volume" for trading volume analysis
- "bar" for dividend history
- "none" for general questions, company info, or when no chart is needed

If user explicitly asks for a specific plot type (scatter, heatmap, etc.), use that type.

Respond in this EXACT format:
CHART_TYPE: [type]
ANSWER: [your well-formatted, structured answer with proper line breaks and bullet points]
"""
        
        response = model.generate_content(context)
        response_text = response.text.strip()
        
        # Parse Gemini response
        chart_type = "none"
        answer = response_text
        
        if "CHART_TYPE:" in response_text and "ANSWER:" in response_text:
            parts = response_text.split("ANSWER:")
            chart_type_part = parts[0].replace("CHART_TYPE:", "").strip().lower()
            answer = parts[1].strip()
            
            # Validate chart type
            valid_types = ["candlestick", "line", "volume", "bar", "none"]
            for vtype in valid_types:
                if vtype in chart_type_part:
                    chart_type = vtype
                    break
        
        return chart_type, answer
        
    except Exception as e:
        print(f"Gemini error: {e}")
        return analyze_without_gemini(question, stock_data)

def analyze_without_gemini(question: str, stock_data: dict):
    """Fallback analysis without Gemini"""
    question_lower = question.lower()
    ticker = stock_data["ticker"]
    info = stock_data["info"]
    hist = stock_data["history"]
    
    if any(word in question_lower for word in ["info", "about", "company", "details", "sector", "business"]):
        answer = f"{info.get('longName', ticker)} operates in the {info.get('sector', 'N/A')} sector. "
        answer += f"Market Cap: ${info.get('marketCap', 0):,.0f}. Current Price: ${info.get('currentPrice', 0):.2f}"
        return "none", answer
    
    elif any(word in question_lower for word in ["price", "chart", "history", "trend", "performance"]):
        if not hist.empty:
            latest_price = hist["Close"].iloc[-1]
            change = ((hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0]) * 100
            answer = f"{ticker} is currently at ${latest_price:.2f}, {change:+.2f}% change in the selected period."
            return "candlestick", answer
        return "none", "No price data available"
    
    elif "dividend" in question_lower:
        dividends = stock_data["dividends"]
        if not dividends.empty:
            answer = f"Latest dividend: ${dividends.iloc[-1]:.2f} on {dividends.index[-1].strftime('%Y-%m-%d')}"
            return "bar", answer
        return "none", f"{ticker} has no dividend history or doesn't pay dividends."
    
    elif "volume" in question_lower:
        if not hist.empty:
            avg_volume = hist["Volume"].mean()
            answer = f"{ticker} average trading volume: {avg_volume:,.0f} shares"
            return "volume", answer
        return "none", "No volume data available"
    
    else:
        if not hist.empty:
            latest_price = hist["Close"].iloc[-1]
            answer = f"{ticker} current price: ${latest_price:.2f}. Ask me about price trends, company info, dividends, or volume!"
            return "line", answer
        return "none", "Please ask about price, company info, dividends, or volume."

def detect_chart_request(question: str):
    """Detect what kind of chart the user wants"""
    q_lower = question.lower()
    
    # Scatter plot
    if "scatter" in q_lower or "correlation" in q_lower or "relationship" in q_lower:
        return "scatter"
    
    # Heatmap
    if "heatmap" in q_lower or "correlation matrix" in q_lower:
        return "heatmap"
    
    # Comparison charts
    if any(word in q_lower for word in ["compare", "vs", "versus", "difference between"]):
        if "performance" in q_lower or "%" in q_lower or "change" in q_lower:
            return "performance_comparison"
        elif "volume" in q_lower:
            return "volume_comparison"
        elif "metric" in q_lower or "p/e" in q_lower or "valuation" in q_lower:
            return "metrics_comparison"
        else:
            return "comparison"
    
    # Single stock charts
    if "candlestick" in q_lower or "ohlc" in q_lower:
        return "candlestick"
    elif "volume" in q_lower:
        return "volume"
    elif "dividend" in q_lower:
        return "bar"
    elif "chart" in q_lower or "graph" in q_lower or "plot" in q_lower or "show" in q_lower or "display" in q_lower:
        return "line"
    
    return "none"


def extract_tickers_from_question(question: str):
    """Extract stock tickers from question"""
    import re
    # Look for patterns like AAPL, MSFT, etc.
    tickers = re.findall(r'\b[A-Z]{1,5}\b', question)
    # Filter out common words
    common_words = ['I', 'A', 'THE', 'AND', 'OR', 'VS', 'PE', 'EPS', 'CEO', 'CFO', 'AI', 'IT']
    return [t for t in tickers if t not in common_words and len(t) <= 5]


def parse_question_enhanced(question: str, ticker: str, period: str, compare_tickers: list = None):
    """Enhanced RAG-enabled question parser with dynamic chart support"""
    
    # Auto-detect tickers from question if not provided
    if not ticker and not compare_tickers:
        detected_tickers = extract_tickers_from_question(question)
        if len(detected_tickers) >= 2:
            compare_tickers = detected_tickers
        elif len(detected_tickers) == 1:
            ticker = detected_tickers[0]
    
    # Check if it's a comparison request
    is_comparison = compare_tickers or any(word in question.lower() for word in ["compare", "vs", "versus"])
    
    # Check if it's a general question
    general_keywords = ["market", "stocks", "which", "better", "what's happening"]
    is_general = any(keyword in question.lower() for keyword in general_keywords) and not ticker and not compare_tickers
    
    if is_comparison and compare_tickers:
        # Handle comparison with dynamic charts
        return handle_comparison_question(question, compare_tickers, period)
    
    if is_general:
        # Handle general market questions with RAG
        return handle_rag_question(question, ticker, compare_tickers, period)
    
    if not ticker:
        return {
            "answer": "Please provide a stock ticker symbol (e.g., AAPL, GOOGL, MSFT) or ask a general market question.",
            "data": {},
            "chart_type": "none"
        }
    
    try:
        # Fetch stock data
        stock_data = get_stock_data(ticker, period)
        
        # Analyze with Gemini + RAG
        chart_type, answer = analyze_with_gemini(question, stock_data)
        
        # Prepare chart data based on chart type
        data_dict = {}
        hist = stock_data["history"]
        
        if chart_type == "candlestick" and not hist.empty:
            data_dict = {
                "dates": hist.index.strftime("%Y-%m-%d").tolist(),
                "open": hist["Open"].tolist(),
                "high": hist["High"].tolist(),
                "low": hist["Low"].tolist(),
                "close": hist["Close"].tolist(),
                "volume": hist["Volume"].tolist()
            }
        elif chart_type == "line" and not hist.empty:
            data_dict = {
                "dates": hist.index.strftime("%Y-%m-%d").tolist(),
                "close": hist["Close"].tolist()
            }
        elif chart_type == "volume" and not hist.empty:
            data_dict = {
                "dates": hist.index.strftime("%Y-%m-%d").tolist(),
                "volume": hist["Volume"].tolist()
            }
        elif chart_type == "bar":
            dividends = stock_data["dividends"]
            if not dividends.empty:
                recent_divs = dividends.tail(10)
                data_dict = {
                    "dates": recent_divs.index.strftime("%Y-%m-%d").tolist(),
                    "dividends": recent_divs.tolist()
                }
        
        return {
            "answer": answer,
            "data": data_dict,
            "chart_type": chart_type
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")


def handle_comparison_question(question: str, compare_tickers: list, period: str = "1mo"):
    """Handle stock comparison questions with dynamic charts"""
    try:
        # Detect chart type from user request
        chart_type = detect_chart_request(question)
        
        # If no specific chart detected but it's a comparison, default to comparison chart
        if chart_type == "none":
            chart_type = "comparison"
        
        # Fetch data for all tickers
        stocks_data = {}
        chart_data = {}
        
        for ticker in compare_tickers[:5]:  # Limit to 5 stocks
            try:
                data = get_stock_data(ticker, period)
                stocks_data[ticker] = data
                
                # Prepare chart data
                hist = data["history"]
                if not hist.empty:
                    chart_data[ticker] = {
                        "dates": hist.index.strftime("%Y-%m-%d").tolist(),
                        "close": hist["Close"].tolist(),
                        "volume": hist["Volume"].tolist() if "Volume" in hist.columns else [],
                        "open": hist["Open"].tolist() if "Open" in hist.columns else [],
                        "high": hist["High"].tolist() if "High" in hist.columns else [],
                        "low": hist["Low"].tolist() if "Low" in hist.columns else []
                    }
                    
                    # Add metrics for comparison
                    info = data["info"]
                    chart_data[ticker].update({
                        "price": info.get("currentPrice", hist["Close"].iloc[-1] if not hist.empty else 0),
                        "pe_ratio": info.get("trailingPE", 0),
                        "market_cap_b": info.get("marketCap", 0) / 1e9,
                        "dividend_yield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0
                    })
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
        
        if not chart_data:
            return {
                "answer": "Unable to fetch data for the requested stocks.",
                "data": {},
                "chart_type": "none"
            }
        
        # Build context for AI
        context = f"""You are comparing these stocks: {', '.join(chart_data.keys())}

Stock Data:
"""
        for ticker, data in chart_data.items():
            context += f"\n{ticker}:\n"
            context += f"- Current Price: ${data['price']:.2f}\n"
            context += f"- P/E Ratio: {data['pe_ratio']:.2f}\n"
            context += f"- Market Cap: ${data['market_cap_b']:.2f}B\n"
            context += f"- Dividend Yield: {data['dividend_yield']:.2f}%\n"
        
        context += f"\nUser Question: {question}\n\n"
        context += """Provide a comprehensive comparison:
1. Brief overview of each stock
2. Key differences and similarities
3. Performance analysis
4. Which might be better for different investor types
5. Use bullet points for clarity"""
        
        if model:
            response = model.generate_content(context)
            answer = format_ai_response(response.text.strip())
        else:
            answer = f"Comparing {', '.join(chart_data.keys())}..."
        
        return {
            "answer": answer,
            "data": chart_data,
            "chart_type": chart_type
        }
        
    except Exception as e:
        return {
            "answer": f"Error processing comparison: {str(e)}",
            "data": {},
            "chart_type": "none"
        }


def handle_rag_question(question: str, ticker: str = None, compare_tickers: list = None, period: str = "1mo"):
    """Handle RAG-enabled questions including comparisons"""
    try:
        # Retrieve relevant context from news
        relevant_context = ""
        if vector_db and vector_db.client:
            try:
                search_query = question
                if ticker:
                    search_query = f"{ticker} {question}"
                if compare_tickers:
                    search_query = f"{' '.join(compare_tickers)} {question}"
                
                relevant_articles = vector_db.hybrid_search("news_articles", search_query, k=3)
                if relevant_articles:
                    relevant_context = "\n\nRelevant Recent News:\n"
                    for idx, article in enumerate(relevant_articles[:3], 1):
                        relevant_context += f"{idx}. {article.get('title', 'N/A')}\n"
                        relevant_context += f"   {article.get('summary', 'N/A')[:200]}\n\n"
            except Exception as e:
                print(f"RAG retrieval error: {e}")
        
        # Fetch stock data for comparison
        stocks_data = {}
        if ticker:
            try:
                stocks_data[ticker] = get_stock_data(ticker, period)
            except:
                pass
        
        if compare_tickers:
            for t in compare_tickers:
                try:
                    stocks_data[t] = get_stock_data(t, period)
                except:
                    pass
        
        # Build comprehensive context
        context = "You are a financial analyst with access to real-time data and news.\n\n"
        
        if stocks_data:
            context += "Stock Data:\n"
            for t, data in stocks_data.items():
                info = data["info"]
                hist = data["history"]
                if not hist.empty:
                    current_price = hist["Close"].iloc[-1]
                    price_change = ((hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0]) * 100
                    context += f"\n{t}:\n"
                    context += f"- Current Price: ${current_price:.2f}\n"
                    context += f"- Change: {price_change:+.2f}%\n"
                    context += f"- Market Cap: ${info.get('marketCap', 0):,.0f}\n"
                    context += f"- P/E Ratio: {info.get('trailingPE', 'N/A')}\n"
                    context += f"- Sector: {info.get('sector', 'N/A')}\n"
        
        context += relevant_context
        context += f"\nUser Question: {question}\n\n"
        context += """Instructions:
1. Provide a clear, well-structured answer
2. Use bullet points (•) for key metrics and comparisons
3. Add blank lines between sections for readability
4. When comparing stocks, create a clear side-by-side comparison
5. Highlight key differences and similarities
6. Be concise but comprehensive

Format your response with:
- Brief overview
- Key metrics (use bullet points)
- Comparison table or side-by-side format (if comparing)
- Analysis and insights
- Recommendation or conclusion (if appropriate)"""
        
        if not model:
            return {
                "answer": "AI model not available",
                "data": {},
                "chart_type": "none"
            }
        
        response = model.generate_content(context)
        answer = format_ai_response(response.text.strip())
        
        return {
            "answer": answer,
            "data": {},
            "chart_type": "none"
        }
        
    except Exception as e:
        return {
            "answer": f"Error processing question: {str(e)}",
            "data": {},
            "chart_type": "none"
        }


def format_ai_response(text: str) -> str:
    """Clean and format AI response for better readability"""
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Add line breaks after sentences for better readability
    text = text.replace('. ', '.\n\n')
    
    # Fix bullet points
    text = text.replace('• ', '\n• ')
    text = text.replace('- ', '\n• ')
    
    # Remove multiple consecutive line breaks
    while '\n\n\n' in text:
        text = text.replace('\n\n\n', '\n\n')
    
    return text.strip()


def parse_question(question: str, ticker: str, period: str):
    """Legacy function - redirects to enhanced version"""
    return parse_question_enhanced(question, ticker, period)

def generate_suggestions(ticker: str, question: str) -> list:
    """Generate follow-up question suggestions"""
    suggestions = [
        f"What's the P/E ratio of {ticker}?",
        f"Show me {ticker}'s dividend history",
        f"How has {ticker} performed this year?",
        f"Compare {ticker} to its 52-week high",
        f"What sector is {ticker} in?",
        "Show me the trading volume"
    ]
    return suggestions[:4]

@app.get("/")
def read_root():
    return {
        "message": "FinancePilot API - Powered by Gemini 2.0 Flash + OpenSearch Vector DB",
        "opensearch_status": "connected" if vector_db and vector_db.client else "disconnected"
    }


@app.post("/search-news")
def search_news(request: dict):
    """Search news articles using vector similarity"""
    try:
        query = request.get("query", "")
        k = request.get("k", 5)
        
        if not vector_db or not vector_db.client:
            return {"error": "OpenSearch not available", "results": []}
        
        results = vector_db.hybrid_search("news_articles", query, k=k)
        return {"results": results, "count": len(results)}
    except Exception as e:
        return {"error": str(e), "results": []}


@app.post("/rag-query")
def rag_query(request: dict):
    """Answer questions using RAG with OpenSearch and Gemini"""
    try:
        question = request.get("question", "")
        if not question:
            return {"error": "No question provided"}
        
        # Retrieve relevant context from OpenSearch
        relevant_articles = []
        if vector_db and vector_db.client:
            try:
                relevant_articles = vector_db.hybrid_search("news_articles", question, k=5)
            except Exception as e:
                print(f"RAG retrieval error: {e}")
        
        # Build context from retrieved articles
        context = "Recent Market News:\n\n"
        if relevant_articles:
            for idx, article in enumerate(relevant_articles[:5], 1):
                context += f"{idx}. {article.get('title', 'N/A')}\n"
                context += f"   {article.get('summary', 'N/A')}\n"
                context += f"   Source: {article.get('source', 'N/A')} | {article.get('published', 'N/A')}\n\n"
        else:
            context = "No recent news articles found in the database.\n\n"
        
        # Use Gemini to generate answer
        if not model:
            return {"answer": "AI model not available", "sources": []}
        
        prompt = f"""You are a knowledgeable financial assistant. Answer the user's question based on the provided context and your knowledge.

{context}

User Question: {question}

Instructions:
1. Provide a comprehensive, conversational answer
2. Reference specific news articles when relevant
3. If the context doesn't contain relevant information, use your general knowledge
4. Be helpful, insightful, and engaging
5. Cite sources when using information from the articles

Answer:"""
        
        response = model.generate_content(prompt)
        answer = response.text.strip()
        
        # Extract sources
        sources = [
            {
                "title": article.get("title"),
                "url": article.get("url"),
                "source": article.get("source")
            }
            for article in relevant_articles[:3]
        ]
        
        return {
            "answer": answer,
            "sources": sources,
            "context_used": len(relevant_articles) > 0
        }
        
    except Exception as e:
        return {"error": str(e), "answer": "Error processing question"}


@app.get("/opensearch-status")
def opensearch_status():
    """Check OpenSearch connection status"""
    if not vector_db or not vector_db.client:
        return {"status": "disconnected", "message": "OpenSearch not available"}
    
    try:
        info = vector_db.client.info()
        return {
            "status": "connected",
            "cluster_name": info.get("cluster_name"),
            "version": info.get("version", {}).get("number"),
            "indices": {
                "news_articles": vector_db.client.indices.exists(index="news_articles"),
                "stock_data": vector_db.client.indices.exists(index="stock_data"),
                "chat_history": vector_db.client.indices.exists(index="chat_history")
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/market-news")
def get_market_news():
    """Get latest market news headlines using SerpAPI with Gemini summaries (cached)"""
    try:
        # Check cache first
        if news_cache["data"] and news_cache["timestamp"]:
            age = (datetime.now() - news_cache["timestamp"]).total_seconds()
            if age < CACHE_TTL:
                print(f"✅ Returning cached news (age: {age:.0f}s)")
                return news_cache["data"]
        
        if not SERPAPI_KEY or SERPAPI_KEY == "your_serpapi_key_here":
            return get_market_news_fallback()
        
        # Use SerpAPI to get stock market news with better parameters
        params = {
            "engine": "google",
            "q": "stock market news",
            "api_key": SERPAPI_KEY,
            "gl": "us",
            "hl": "en",
            "tbm": "nws",  # News search
            "num": 10
        }
        
        response = requests.get("https://serpapi.com/search", params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            news_results = data.get("news_results", [])
            
            if not news_results:
                print("No news results from SerpAPI, using fallback")
                return get_market_news_fallback()
            
            all_news = []
            articles_to_summarize = []
            
            # First pass: collect articles and use snippets directly when good
            for idx, article in enumerate(news_results[:10]):
                source_info = article.get("source", {})
                if isinstance(source_info, dict):
                    source_name = source_info.get("name", "Unknown")
                else:
                    source_name = str(source_info) if source_info else "Unknown"
                
                title = article.get("title", "No title")
                snippet = article.get("snippet", article.get("description", article.get("highlight", {}).get("snippet", "")))
                
                # Use snippet directly if it's good enough (>50 chars)
                if snippet and len(snippet) > 50:
                    summary = snippet
                else:
                    # Mark for Gemini processing
                    articles_to_summarize.append((idx, title, snippet))
                    summary = snippet if snippet else f"Latest update: {title}"
                
                news_item = {
                    "title": title,
                    "summary": summary,
                    "url": article.get("link", "#"),
                    "published": article.get("date", "Recently"),
                    "source": source_name,
                    "type": "news",
                    "index": idx
                }
                all_news.append(news_item)
            
            # Second pass: Batch process with Gemini only if needed (max 3 articles)
            if model and articles_to_summarize and len(articles_to_summarize) <= 3:
                try:
                    # Batch prompt for multiple articles
                    batch_prompt = "Provide brief 1-sentence summaries for these news headlines:\n\n"
                    for idx, title, snippet in articles_to_summarize:
                        batch_prompt += f"{idx+1}. {title}\n"
                    batch_prompt += "\nProvide numbered summaries (1., 2., etc.):"
                    
                    gemini_response = model.generate_content(batch_prompt)
                    summaries_text = gemini_response.text.strip()
                    
                    # Parse numbered responses
                    for idx, title, snippet in articles_to_summarize:
                        # Try to extract the specific summary
                        import re
                        pattern = rf"{idx+1}\.\s*(.+?)(?=\n\d+\.|$)"
                        match = re.search(pattern, summaries_text, re.DOTALL)
                        if match:
                            summary = match.group(1).strip()
                            # Update the news item
                            for item in all_news:
                                if item["index"] == idx:
                                    item["summary"] = summary
                                    break
                except Exception as e:
                    print(f"Batch Gemini error: {e}")
                
                # Skip OpenSearch indexing to avoid timeouts
                # Can be re-enabled when OpenSearch is properly set up
            
            print(f"✅ Fetched {len(all_news)} news articles from SerpAPI")
            print(f"✅ Stored {len(all_news)} articles in OpenSearch vector DB")
            
            # Cache the result
            result = {"news": all_news}
            news_cache["data"] = result
            news_cache["timestamp"] = datetime.now()
            
            return result
        else:
            print(f"SerpAPI returned status {response.status_code}, using fallback")
            return get_market_news_fallback()
            
    except Exception as e:
        print(f"SerpAPI news fetch error: {e}")
        return get_market_news_fallback()


def get_market_news_fallback():
    """Fallback method using yfinance"""
    try:
        import yfinance as yf
        from datetime import datetime
        
        # Get news from major market tickers
        news_sources = ["^GSPC", "^DJI", "^IXIC"]
        all_news = []
        
        for ticker in news_sources:
            try:
                stock = yf.Ticker(ticker)
                news = stock.news
                
                for article in news[:5]:
                    all_news.append({
                        "title": article.get("title", "No title"),
                        "summary": article.get("summary", "No summary available"),
                        "url": article.get("link", "#"),
                        "published": datetime.fromtimestamp(article.get("providerPublishTime", 0)).strftime("%Y-%m-%d %H:%M"),
                        "source": article.get("publisher", "Unknown")
                    })
            except:
                continue
        
        # Remove duplicates
        seen_titles = set()
        unique_news = []
        for article in all_news:
            if article["title"] not in seen_titles:
                seen_titles.add(article["title"])
                unique_news.append(article)
        
        unique_news.sort(key=lambda x: x["published"], reverse=True)
        return {"news": unique_news[:10]}
    except Exception as e:
        print(f"Fallback news fetch error: {e}")
        return {"news": []}


@app.post("/fetch-article")
def fetch_article(request: dict):
    """Fetch full article content from URL and summarize with Gemini"""
    try:
        url = request.get("url")
        if not url or url == "#":
            return {"error": "Invalid URL", "content": ""}
        
        # Try using newspaper3k to extract article
        try:
            from newspaper import Article
            
            article = Article(url)
            article.download()
            article.parse()
            
            full_text = article.text
            title = article.title
            
            if not full_text:
                return {"error": "Could not extract article content", "content": ""}
            
            # Use Gemini to create a comprehensive summary
            if model:
                try:
                    prompt = f"""You are a financial news analyst. Provide a comprehensive summary of this article for stock market investors.

Title: {title}

Article Content:
{full_text[:4000]}

Create a detailed summary that includes:
1. Main points and key takeaways
2. Market implications
3. Important data or statistics mentioned
4. What investors should know

Format the summary in clear paragraphs."""
                    
                    gemini_response = model.generate_content(prompt)
                    summary = gemini_response.text.strip()
                    
                    return {
                        "title": title,
                        "content": summary,
                        "full_text": full_text[:2000],  # First 2000 chars of original
                        "success": True
                    }
                except Exception as e:
                    print(f"Gemini summarization error: {e}")
                    return {
                        "title": title,
                        "content": full_text[:1500],
                        "success": True
                    }
            else:
                return {
                    "title": title,
                    "content": full_text[:1500],
                    "success": True
                }
                
        except Exception as e:
            print(f"Article extraction error: {e}")
            return {"error": f"Could not fetch article: {str(e)}", "content": "", "success": False}
            
    except Exception as e:
        print(f"Fetch article error: {e}")
        return {"error": str(e), "content": "", "success": False}


@app.get("/market-overview")
def get_market_overview():
    """Get top gainers, losers, and most active stocks"""
    try:
        # Popular stocks to check
        tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", 
            "NFLX", "DIS", "PYPL", "INTC", "COIN", "SNAP", "PLTR", "RIVN",
            "LCID", "NIO", "BABA", "JD", "PFE", "MRNA", "BA", "GE", "F"
        ]
        
        def get_stock_summary(ticker):
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="2d")
                
                if len(hist) < 2:
                    return None
                
                current_price = hist["Close"].iloc[-1]
                prev_close = hist["Close"].iloc[-2]
                change = current_price - prev_close
                change_pct = (change / prev_close) * 100 if prev_close else 0
                
                # Get company name
                try:
                    info = stock.info
                    name = info.get("shortName", ticker)
                    volume = hist["Volume"].iloc[-1]
                except:
                    name = ticker
                    volume = hist["Volume"].iloc[-1] if "Volume" in hist.columns else 0
                
                return {
                    "ticker": ticker,
                    "name": name,
                    "price": round(float(current_price), 2),
                    "change": round(float(change), 2),
                    "change_pct": round(float(change_pct), 2),
                    "volume": int(volume)
                }
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
                return None
        
        # Fetch all stocks
        all_stocks = [s for s in [get_stock_summary(t) for t in tickers] if s]
        
        # Sort and categorize
        gainers = sorted([s for s in all_stocks if s["change_pct"] > 0], 
                        key=lambda x: x["change_pct"], reverse=True)[:5]
        losers = sorted([s for s in all_stocks if s["change_pct"] < 0], 
                       key=lambda x: x["change_pct"])[:5]
        active = sorted(all_stocks, key=lambda x: x["volume"], reverse=True)[:5]
        
        return {
            "gainers": gainers,
            "losers": losers,
            "active": active
        }
    except Exception as e:
        print(f"Market overview error: {e}")
        return {
            "gainers": [],
            "losers": [],
            "active": []
        }

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """General chat endpoint for any question"""
    question = request.question.lower()
    
    # Check if user is asking a general question without context
    if not model:
        return ChatResponse(
            answer="I'm a stock market assistant. Please provide a stock ticker (like AAPL, GOOGL, MSFT) to get started!",
            needs_ticker=True,
            suggestions=["Tell me about AAPL", "Show me TSLA stock", "What's GOOGL doing?"]
        )
    
    try:
        # Use Gemini for general conversation
        context = f"""
You are a friendly and knowledgeable stock market assistant. The user asked: "{request.question}"

If the question is about:
1. A specific stock - ask them to provide the ticker symbol
2. General investing advice - provide helpful, educational information
3. How to use the chatbot - explain features
4. Market concepts - explain clearly and simply

Be conversational, helpful, and engaging. Keep responses concise (2-3 sentences).

If you need a stock ticker to answer, say so clearly.
"""
        
        response = model.generate_content(context)
        answer = response.text.strip()
        
        # Check if we need a ticker
        needs_ticker = any(word in answer.lower() for word in ["ticker", "symbol", "which stock", "what stock"])
        
        suggestions = [
            "Tell me about AAPL",
            "Show me TSLA performance",
            "What's GOOGL's market cap?",
            "Explain P/E ratio"
        ]
        
        return ChatResponse(
            answer=answer,
            needs_ticker=needs_ticker,
            suggestions=suggestions
        )
    except Exception as e:
        return ChatResponse(
            answer="I'm here to help you with stock market data! Try asking about a specific stock like AAPL, GOOGL, or TSLA.",
            needs_ticker=True,
            suggestions=["Tell me about AAPL", "Show me TSLA stock", "What's GOOGL doing?"]
        )

@app.post("/query", response_model=QueryResponse)
def query_stock(request: QueryRequest):
    """Process natural language query about stocks with RAG (supports comparisons)"""
    result = parse_question_enhanced(
        request.question, 
        request.ticker, 
        request.period,
        request.compare_tickers
    )
    
    # Add suggestions
    if request.ticker:
        result["suggestions"] = generate_suggestions(request.ticker, request.question)
    
    # Save to NocoDB
    save_to_nocodb(request.question, request.ticker or "", result)
    
    # Skip chat history storage to avoid timeouts
    # Can be re-enabled when OpenSearch is properly set up
    
    return result

@app.get("/history")
def get_history():
    """Get query history from NocoDB"""
    if not NOCODB_TOKEN or not NOCODB_TABLE_ID:
        return {"message": "NocoDB not configured", "data": []}
    
    try:
        headers = {"xc-token": NOCODB_TOKEN}
        response = requests.get(
            f"{NOCODB_URL}/api/v1/db/data/noco/{NOCODB_TABLE_ID}",
            headers=headers
        )
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching history: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
