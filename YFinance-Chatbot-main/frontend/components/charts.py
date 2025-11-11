"""Chart creation functions"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_comparison_chart(data):
    """Create comparison chart for multiple stocks"""
    fig = go.Figure()
    
    colors = ['#00ff00', '#ff6b6b', '#4ecdc4', '#ffe66d', '#a8dadc']
    
    for idx, (ticker, stock_data) in enumerate(data.items()):
        fig.add_trace(go.Scatter(
            x=stock_data["dates"],
            y=stock_data["close"],
            mode="lines",
            name=ticker,
            line=dict(color=colors[idx % len(colors)], width=2)
        ))
    
    fig.update_layout(
        title="Stock Price Comparison",
        yaxis_title="Price (USD)",
        xaxis_title="Date",
        height=500,
        template="plotly_dark",
        hovermode='x unified'
    )
    
    return fig


def create_performance_comparison(data):
    """Create normalized performance comparison (percentage change)"""
    fig = go.Figure()
    
    colors = ['#00ff00', '#ff6b6b', '#4ecdc4', '#ffe66d', '#a8dadc']
    
    for idx, (ticker, stock_data) in enumerate(data.items()):
        # Normalize to percentage change from first value
        closes = stock_data["close"]
        if closes and len(closes) > 0:
            first_price = closes[0]
            normalized = [(price / first_price - 1) * 100 for price in closes]
            
            fig.add_trace(go.Scatter(
                x=stock_data["dates"],
                y=normalized,
                mode="lines",
                name=ticker,
                line=dict(color=colors[idx % len(colors)], width=2)
            ))
    
    fig.update_layout(
        title="Performance Comparison (% Change)",
        yaxis_title="Change (%)",
        xaxis_title="Date",
        height=500,
        template="plotly_dark",
        hovermode='x unified'
    )
    
    return fig


def create_volume_comparison(data):
    """Create volume comparison chart"""
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, (ticker, stock_data) in enumerate(data.items()):
        fig.add_trace(go.Bar(
            x=stock_data["dates"],
            y=stock_data["volume"],
            name=ticker,
            marker_color=colors[idx % len(colors)],
            opacity=0.7
        ))
    
    fig.update_layout(
        title="Volume Comparison",
        yaxis_title="Volume",
        xaxis_title="Date",
        height=500,
        template="plotly_dark",
        barmode='group'
    )
    
    return fig


def create_metrics_comparison(data):
    """Create bar chart comparing key metrics"""
    fig = go.Figure()
    
    tickers = list(data.keys())
    metrics = ['price', 'pe_ratio', 'market_cap_b', 'dividend_yield']
    metric_names = ['Current Price ($)', 'P/E Ratio', 'Market Cap (B)', 'Dividend Yield (%)']
    
    for idx, metric in enumerate(metrics):
        values = [data[ticker].get(metric, 0) for ticker in tickers]
        
        fig.add_trace(go.Bar(
            name=metric_names[idx],
            x=tickers,
            y=values,
            text=[f'{v:.2f}' for v in values],
            textposition='auto',
        ))
    
    fig.update_layout(
        title="Key Metrics Comparison",
        yaxis_title="Value",
        xaxis_title="Stock",
        height=500,
        template="plotly_dark",
        barmode='group'
    )
    
    return fig


def create_scatter_plot(data):
    """Create scatter plot comparing two stocks"""
    fig = go.Figure()
    
    tickers = list(data.keys())
    if len(tickers) >= 2:
        ticker1, ticker2 = tickers[0], tickers[1]
        
        # Get price data
        prices1 = data[ticker1].get("close", [])
        prices2 = data[ticker2].get("close", [])
        
        # Ensure same length
        min_len = min(len(prices1), len(prices2))
        prices1 = prices1[:min_len]
        prices2 = prices2[:min_len]
        
        fig.add_trace(go.Scatter(
            x=prices1,
            y=prices2,
            mode='markers',
            marker=dict(
                size=8,
                color=list(range(len(prices1))),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Time")
            ),
            text=[f"Day {i+1}" for i in range(len(prices1))],
            hovertemplate=f'{ticker1}: $%{{x:.2f}}<br>{ticker2}: $%{{y:.2f}}<br>%{{text}}<extra></extra>'
        ))
        
        # Add trend line
        if len(prices1) > 1:
            import numpy as np
            z = np.polyfit(prices1, prices2, 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=prices1,
                y=p(prices1),
                mode='lines',
                name='Trend',
                line=dict(color='red', dash='dash')
            ))
        
        fig.update_layout(
            title=f"Scatter Plot: {ticker1} vs {ticker2}",
            xaxis_title=f"{ticker1} Price ($)",
            yaxis_title=f"{ticker2} Price ($)",
            height=500,
            template="plotly_dark",
            hovermode='closest'
        )
    
    return fig


def create_correlation_heatmap(data):
    """Create correlation heatmap for multiple stocks"""
    import pandas as pd
    
    # Build dataframe
    df_data = {}
    for ticker, stock_data in data.items():
        df_data[ticker] = stock_data.get("close", [])
    
    df = pd.DataFrame(df_data)
    correlation = df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation.values,
        x=correlation.columns,
        y=correlation.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 12},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Stock Price Correlation Matrix",
        height=500,
        template="plotly_dark"
    )
    
    return fig


def create_custom_chart(data, chart_config):
    """Create custom chart based on configuration"""
    chart_type = chart_config.get("type", "scatter")
    
    if chart_type == "scatter":
        return create_scatter_plot(data)
    elif chart_type == "heatmap":
        return create_correlation_heatmap(data)
    else:
        return create_comparison_chart(data)


def create_candlestick_chart(data):
    """Create candlestick chart with Plotly"""
    fig = go.Figure(data=[go.Candlestick(
        x=data["dates"],
        open=data["open"],
        high=data["high"],
        low=data["low"],
        close=data["close"],
        name="Price"
    )])
    
    fig.update_layout(
        title="Stock Price Chart",
        yaxis_title="Price (USD)",
        xaxis_title="Date",
        height=500,
        template="plotly_dark"
    )
    
    return fig


def create_line_chart(data):
    """Create line chart with Plotly"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data["dates"],
        y=data["close"],
        mode="lines",
        name="Close Price",
        line=dict(color="#00ff00", width=2)
    ))
    
    fig.update_layout(
        title="Stock Price Trend",
        yaxis_title="Price (USD)",
        xaxis_title="Date",
        height=500,
        template="plotly_dark"
    )
    
    return fig


def create_volume_chart(data):
    """Create volume bar chart with Plotly"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=data["dates"],
        y=data["volume"],
        name="Volume",
        marker_color="#1f77b4"
    ))
    
    fig.update_layout(
        title="Trading Volume",
        yaxis_title="Volume",
        xaxis_title="Date",
        height=500,
        template="plotly_dark"
    )
    
    return fig


def create_dividend_chart(data):
    """Create dividend bar chart with Plotly"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=data["dates"],
        y=data["dividends"],
        name="Dividends",
        marker_color="#ff7f0e"
    ))
    
    fig.update_layout(
        title="Dividend History",
        yaxis_title="Dividend (USD)",
        xaxis_title="Date",
        height=500,
        template="plotly_dark"
    )
    
    return fig
