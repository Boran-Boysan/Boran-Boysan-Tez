import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import register_page, html, dcc, Input, Output, State, callback
import dash.exceptions
import numpy as np

register_page(__name__, path="/stok", name="Stok Yönetimi ve Analizi")

# Veri Yükleme
df = pd.read_csv("demand_forecasting_data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df[df["Demand"].notna() & (df["Demand"] > 0)].copy()
df["Year"] = df["Date"].dt.year
df["Week"] = df["Date"].dt.isocalendar().week
df["Month"] = df["Date"].dt.month
df["Quarter"] = df["Date"].dt.quarter
df["Public_Holiday"] = df["Public_Holiday"].astype(int)
df["Seasonal_Trend"] = df["Seasonal_Trend"].astype("category")
df["Marketing_Campaign"] = df["Marketing_Campaign"].astype("category")
df["Product_ID"] = df["Product_ID"].astype("category")

# Stok durumu kategorileri oluştur
df['Stock_Category'] = pd.cut(df['Stock_Availability'],
                              bins=[0, 30, 70, 90, 100],
                              labels=['🔴 Kritik', '🟡 Düşük', '🟢 Normal', '🔵 Yüksek'],
                              include_lowest=True)

# Stok-talep oranı hesapla
df['Stock_Demand_Ratio'] = df['Stock_Availability'] / (df['Demand'] + 1)  # +1 to avoid division by zero
df['Stock_Efficiency'] = np.where(df['Stock_Demand_Ratio'] > 2, 'Fazla Stok',
                                  np.where(df['Stock_Demand_Ratio'] < 0.5, 'Stok Yetersizliği', 'Optimal'))

# Stok devir hızı (basit hesaplama)
df['Stock_Turnover'] = df['Demand'] / (df['Stock_Availability'] + 1)

PLOT_BG_COLOR = '#ECF0F1'
PRIMARY_COLOR = '#2C3E50'
ACCENT_COLOR = '#E67E22'
SUCCESS_COLOR = '#27AE60'
WARNING_COLOR = '#F39C12'
DANGER_COLOR = '#E74C3C'

# Grafik kutu stili - app.py ile uyumlu
graph_style = {
    'boxShadow': '0 8px 32px rgba(0,0,0,0.1)',
    'borderRadius': '15px',
    'marginBottom': '30px',
    'backgroundColor': '#fff',
    'overflow': 'hidden',
    'transition': 'all 0.3s ease',
    'border': '1px solid #e0e0e0'
}


def layout():
    return html.Div([
        html.H3("📦 Gelişmiş Stok Yönetimi ve Analizi", style={
            'textAlign': 'center',
            'color': PRIMARY_COLOR,
            'marginBottom': '30px',
            'fontSize': '2rem',
            'fontWeight': 'bold'
        }),

        # Stok durumu özet kartları
        html.Div([
            html.Div([
                html.H4("📊 Ortalama Stok Seviyesi", style={'textAlign': 'center', 'margin': '0'}),
                html.H2(id="avg-stock-level", style={'textAlign': 'center', 'color': '#fff', 'margin': '10px 0'})
            ], style={
                'background': f'linear-gradient(135deg, {ACCENT_COLOR}, #D35400)',
                'padding': '20px',
                'borderRadius': '10px',
                'color': 'white',
                'width': '30%',
                'margin': '15px'
            }),

            html.Div([
                html.H4("⚡ Ortalama Devir Hızı", style={'textAlign': 'center', 'margin': '0'}),
                html.H2(id="avg-turnover", style={'textAlign': 'center', 'color': '#fff', 'margin': '10px 0'})
            ], style={
                'background': f'linear-gradient(135deg, {SUCCESS_COLOR}, #229954)',
                'padding': '20px',
                'borderRadius': '10px',
                'color': 'white',
                'width': '30%',
                'margin': '15px'
            }),

            html.Div([
                html.H4("🔴 Kritik Stok Ürünleri", style={'textAlign': 'center', 'margin': '0'}),
                html.H2(id="critical-products", style={'textAlign': 'center', 'color': '#fff', 'margin': '10px 0'})
            ], style={
                'background': f'linear-gradient(135deg, {DANGER_COLOR}, #C0392B)',
                'padding': '20px',
                'borderRadius': '10px',
                'color': 'white',
                'width': '30%',
                'margin': '15px'
            })
        ], style={'display': 'flex', 'justifyContent': 'center', 'flexWrap': 'wrap', 'marginBottom': '30px'}),

        # Grafik bölümleri - app.py formatı ile uyumlu
        html.Div([
            html.Div("📈 Stok Seviyesi ve Talep Zaman Serisi", className='graph-header'),
            html.Div([dcc.Graph(id='stock-timeline-graph')], className='graph-content')
        ], className='graph-container'),

        html.Div([
            html.Div("📊 Stok Kategorisi Dağılımı", className='graph-header'),
            html.Div([dcc.Graph(id='stock-category-graph')], className='graph-content')
        ], className='graph-container'),

        html.Div([
            html.Div("📈 Stok Seviyesi vs Talep Korelasyonu", className='graph-header'),
            html.Div([dcc.Graph(id='stock-demand-correlation')], className='graph-content')
        ], className='graph-container'),

        html.Div([
            html.Div("⚡ Ürün Bazında Stok Devir Hızı", className='graph-header'),
            html.Div([dcc.Graph(id='stock-turnover-graph')], className='graph-content')
        ], className='graph-container'),
    ])


@callback(
    Output("stock-timeline-graph", "figure"),
    Output("stock-category-graph", "figure"),
    Output("stock-demand-correlation", "figure"),
    Output("stock-turnover-graph", "figure"),
    Output("avg-stock-level", "children"),
    Output("avg-turnover", "children"),
    Output("critical-products", "children"),
    Input("apply-button", "n_clicks"),
    State("product-dropdown", "value"),
    State("marketing-dropdown", "value"),
    State("price-slider", "value"),
    State("discount-slider", "value"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
    State("url", "pathname")
)
def update_stock_graphs(n, product_id, marketing, price_range, discount_range, start_date, end_date, pathname):
    if pathname != "/stok":
        raise dash.exceptions.PreventUpdate

    # Veri filtreleme
    filtered = df.copy()
    if product_id:
        filtered = filtered[filtered["Product_ID"].isin(product_id)]
    if marketing:
        filtered = filtered[filtered["Marketing_Campaign"].isin(marketing)]
    if price_range:
        filtered = filtered[(filtered["Price"] >= price_range[0]) & (filtered["Price"] <= price_range[1])]
    if discount_range:
        filtered = filtered[
            (filtered["Discount"] >= discount_range[0] / 100) & (filtered["Discount"] <= discount_range[1] / 100)]
    if start_date and end_date:
        filtered = filtered[(filtered["Date"] >= start_date) & (filtered["Date"] <= end_date)]

    # 1. Stok Seviyesi Zaman Serisi
    daily_stock = filtered.groupby('Date').agg({
        'Stock_Availability': 'mean',
        'Demand': 'sum',
        'Product_ID': 'nunique'
    }).reset_index()

    stock_timeline = make_subplots(
        rows=2, cols=1,
        subplot_titles=('📈 Günlük Ortalama Stok Seviyesi', '📊 Günlük Toplam Talep'),
        vertical_spacing=0.1
    )

    stock_timeline.add_trace(
        go.Scatter(
            x=daily_stock['Date'],
            y=daily_stock['Stock_Availability'],
            mode='lines+markers',
            name='Ortalama Stok',
            line=dict(color=ACCENT_COLOR, width=3),
            fill='tozeroy',
            fillcolor=f'rgba(230,126,34,0.3)',
            hovertemplate='<b>Tarih:</b> %{x|%d.%m.%Y}<br><b>Stok:</b> %{y:.1f}%<extra></extra>'
        ),
        row=1, col=1
    )

    stock_timeline.add_trace(
        go.Bar(
            x=daily_stock['Date'],
            y=daily_stock['Demand'],
            name='Toplam Talep',
            marker_color=SUCCESS_COLOR,
            opacity=0.7,
            hovertemplate='<b>Tarih:</b> %{x|%d.%m.%Y}<br><b>Talep:</b> %{y:,.0f}<extra></extra>'
        ),
        row=2, col=1
    )

    stock_timeline.update_layout(
        title={
            'text': "📈 Stok Seviyesi ve Talep Zaman Serisi",
            'font': {'size': 24, 'color': PRIMARY_COLOR},
            'x': 0.5,
            'xanchor': 'center'
        },
        plot_bgcolor=PLOT_BG_COLOR,
        paper_bgcolor="#FFFFFF",
        font=dict(family="Arial", size=12),
        height=600,
        showlegend=True
    )

    # Alt grafik başlıklarını büyüt
    stock_timeline.update_annotations(font_size=18)

    # 2. Stok Kategorisi Dağılımı
    stock_category_data = filtered['Stock_Category'].value_counts().reset_index()
    stock_category_data.columns = ['Kategori', 'Adet']

    # Renkleri kategorilere göre eşleştir
    color_mapping = {
        '🔴 Kritik': '#E74C3C',  # Kırmızı
        '🟡 Düşük': '#F39C12',  # Turuncu
        '🟢 Normal': '#27AE60',  # Yeşil
        '🔵 Yüksek': '#3498DB'  # Mavi
    }

    # Kategorilere göre renkleri sırala
    colors = [color_mapping.get(cat, '#95A5A6') for cat in stock_category_data['Kategori']]

    stock_category = go.Figure(data=[go.Pie(
        labels=stock_category_data['Kategori'],
        values=stock_category_data['Adet'],
        hole=0.4,
        marker=dict(colors=colors, line=dict(color='white', width=2)),
        textinfo='label+percent',
        textposition='outside',
        textfont=dict(size=14),
        hovertemplate='<b>%{label}</b><br>Miktar: %{value:,.0f}<br>Oran: %{percent}<extra></extra>',
        pull=[0.05 if '🔵' not in cat else 0.1 for cat in stock_category_data['Kategori']]
    )])

    stock_category.update_layout(
        title={
            'text': "📊 Stok Kategorisi Dağılımı",
            'font': {'size': 24, 'color': PRIMARY_COLOR},
            'x': 0.5,
            'xanchor': 'center'
        },
        font=dict(family="Arial", size=12),
        paper_bgcolor="#FFFFFF",
        margin=dict(t=100, b=40, l=40, r=40)
    )

    # 3. Stok-Talep Korelasyonu
    correlation_data = filtered.groupby('Product_ID').agg({
        'Stock_Availability': 'mean',
        'Demand': 'mean',
        'Price': 'mean'
    }).reset_index()

    stock_demand_corr = px.scatter(
        correlation_data,
        x='Stock_Availability',
        y='Demand',
        size='Price',
        color='Product_ID',
        title="📈 Stok Seviyesi vs Talep Korelasyonu",
        labels={
            'Stock_Availability': 'Ortalama Stok Seviyesi (%)',
            'Demand': 'Ortalama Talep',
            'Price': 'Ortalama Fiyat'
        },
        hover_data=['Product_ID']
    )

    # Trend çizgisi ekle
    stock_demand_corr.add_trace(
        px.scatter(correlation_data, x='Stock_Availability', y='Demand', trendline="ols").data[1]
    )

    stock_demand_corr.update_layout(
        title={
            'text': "📈 Stok Seviyesi vs Talep Korelasyonu",
            'font': {'size': 24, 'color': PRIMARY_COLOR},
            'x': 0.5,
            'xanchor': 'center'
        },
        plot_bgcolor=PLOT_BG_COLOR,
        paper_bgcolor="#FFFFFF",
        font=dict(family="Arial", size=12)
    )

    # 4. Stok Devir Hızı Analizi
    turnover_by_product = filtered.groupby('Product_ID')['Stock_Turnover'].mean().reset_index()
    turnover_by_product = turnover_by_product.sort_values('Stock_Turnover', ascending=True).tail(15)

    stock_turnover = px.bar(
        turnover_by_product,
        x='Stock_Turnover',
        y='Product_ID',
        orientation='h',
        title="⚡ Ürün Bazında Stok Devir Hızı (En Yüksek 15)",
        labels={'Stock_Turnover': 'Stok Devir Hızı', 'Product_ID': 'Ürün ID'},
        color='Stock_Turnover',
        color_continuous_scale='RdYlGn'
    )

    stock_turnover.update_layout(
        title={
            'text': "⚡ Ürün Bazında Stok Devir Hızı (En Yüksek 15)",
            'font': {'size': 24, 'color': PRIMARY_COLOR},
            'x': 0.5,
            'xanchor': 'center'
        },
        plot_bgcolor=PLOT_BG_COLOR,
        paper_bgcolor="#FFFFFF",
        font=dict(family="Arial", size=12),
        height=500
    )

    # KPI hesaplamaları
    avg_stock = f"{filtered['Stock_Availability'].mean():.1f}%"
    avg_turnover_val = f"{filtered['Stock_Turnover'].mean():.2f}"
    critical_count = len(filtered[filtered['Stock_Category'] == '🔴 Kritik'])

    return (
        stock_timeline,
        stock_category,
        stock_demand_corr,
        stock_turnover,
        avg_stock,
        avg_turnover_val,
        str(critical_count)
    )