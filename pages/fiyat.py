import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import register_page, html, dcc, Input, Output, State, callback
import dash.exceptions
import numpy as np

register_page(__name__, path="/fiyat", name="Fiyat Analizi ve Strateji")

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

# Basit metrikler
df['Revenue'] = df['Price'] * df['Demand']
df['Price_Difference'] = df['Price'] - df['Competitor_Price']

# Fiyat kategorileri
df['Price_Category'] = pd.cut(df['Price'],
                              bins=[0, 50, 100, 200, float('inf')],
                              labels=['💚 Düşük', '💛 Orta', '🧡 Yüksek', '🔴 Premium'],
                              include_lowest=True)

PLOT_BG_COLOR = '#ECF0F1'
PRIMARY_COLOR = '#2C3E50'
ACCENT_COLOR = '#E67E22'
SUCCESS_COLOR = '#27AE60'
WARNING_COLOR = '#F39C12'
DANGER_COLOR = '#E74C3C'


def layout():
    return html.Div([
        html.H3("💰 Fiyat Analizi ve Rekabet Stratejisi", style={
            'textAlign': 'center',
            'color': PRIMARY_COLOR,
            'marginBottom': '30px',
            'fontSize': '2rem',
            'fontWeight': 'bold'
        }),

        # KPI kartları
        html.Div([
            html.Div([
                html.H4("💵 Ortalama Fiyat", style={'textAlign': 'center', 'margin': '0'}),
                html.H2(id="price-avg-price", style={'textAlign': 'center', 'color': '#fff', 'margin': '10px 0'})
            ], className='kpi-card', style={
                'background': f'linear-gradient(135deg, {ACCENT_COLOR}, #D35400)',
                'padding': '20px',
                'borderRadius': '10px',
                'color': 'white',
                'width': '30%',
                'margin': '15px'
            }),

            html.Div([
                html.H4("🏆 En Karlı Ürün", style={'textAlign': 'center', 'margin': '0'}),
                html.H2(id="price-most-profitable", style={'textAlign': 'center', 'color': '#fff', 'margin': '10px 0'})
            ], className='kpi-card', style={
                'background': f'linear-gradient(135deg, {SUCCESS_COLOR}, #229954)',
                'padding': '20px',
                'borderRadius': '10px',
                'color': 'white',
                'width': '30%',
                'margin': '15px'
            }),

            html.Div([
                html.H4("📊 Toplam Gelir", style={'textAlign': 'center', 'margin': '0'}),
                html.H2(id="price-total-revenue", style={'textAlign': 'center', 'color': '#fff', 'margin': '10px 0'})
            ], className='kpi-card', style={
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
            html.Div("💰 Fiyat vs Talep İlişkisi", className='graph-header'),
            html.Div([dcc.Graph(id='price-demand-scatter')], className='graph-content')
        ], className='graph-container'),

        html.Div([
            html.Div("🎯 Fiyat Kategorisi Satış Huni Analizi", className='graph-header'),
            html.Div([dcc.Graph(id='price-funnel-chart')], className='graph-content')
        ], className='graph-container'),

        html.Div([
            html.Div("🎪 3D Fiyat-İndirim-Talep Yüzey Analizi", className='graph-header'),
            html.Div([dcc.Graph(id='price-3d-surface')], className='graph-content')
        ], className='graph-container'),

        html.Div([
            html.Div("🆚 Rakip Fiyat Karşılaştırması", className='graph-header'),
            html.Div([dcc.Graph(id='competitor-comparison')], className='graph-content')
        ], className='graph-container'),

        html.Div([
            html.Div("📈 Haftalık Fiyat Trendi", className='graph-header'),
            html.Div([dcc.Graph(id='price-trend-line')], className='graph-content')
        ], className='graph-container'),
    ])


@callback(
    Output("price-demand-scatter", "figure"),
    Output("price-funnel-chart", "figure"),  # YENİ: Funnel Chart output
    Output("price-3d-surface", "figure"),  # YENİ: 3D Surface output
    Output("competitor-comparison", "figure"),
    Output("price-trend-line", "figure"),
    Output("price-avg-price", "children"),
    Output("price-most-profitable", "children"),
    Output("price-total-revenue", "children"),
    Input("apply-button", "n_clicks"),
    State("product-dropdown", "value"),
    State("marketing-dropdown", "value"),
    State("price-slider", "value"),
    State("discount-slider", "value"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
    State("url", "pathname")
)
def update_price_graphs(n, product_id, marketing, price_range, discount_range, start_date, end_date, pathname):
    if pathname != "/fiyat":
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

    # 1. Fiyat-Talep Scatter Grafiği (mevcut)
    sample_data = filtered.sample(min(500, len(filtered))) if len(filtered) > 500 else filtered

    scatter_fig = px.scatter(
        sample_data,
        x='Price',
        y='Demand',
        color='Marketing_Campaign',
        size='Revenue',
        title="💰 Fiyat vs Talep İlişkisi",
        labels={'Price': 'Fiyat (₺)', 'Demand': 'Talep'},
        hover_data=['Product_ID', 'Revenue']
    )

    scatter_fig.update_layout(
        title={
            'text': "💰 Fiyat vs Talep İlişkisi",
            'font': {'size': 24, 'color': PRIMARY_COLOR},
            'x': 0.5,
            'xanchor': 'center'
        },
        plot_bgcolor=PLOT_BG_COLOR,
        paper_bgcolor="#FFFFFF",
        font=dict(family="Arial", size=12)
    )

    # 2. YENİ: Funnel Chart - Fiyat Kategorilerinden Satış Dönüşümü
    # Sadece mevcut verideki kategorileri kullan ve 3'e böl
    available_categories = filtered['Price_Category'].dropna().unique()

    # Mevcut fiyat verilerini 3 kategoriye böl
    price_min, price_max = filtered['Price'].min(), filtered['Price'].max()
    price_range = price_max - price_min

    # 3 eşit kategoriye böl
    low_threshold = price_min + (price_range / 3)
    high_threshold = price_min + (2 * price_range / 3)

    # Yeni 3'lü kategori oluştur
    filtered['Funnel_Category'] = pd.cut(
        filtered['Price'],
        bins=[price_min - 0.1, low_threshold, high_threshold, price_max + 0.1],
        labels=['💚 Düşük Fiyat', '💛 Orta Fiyat', '🧡 Yüksek Fiyat'],
        include_lowest=True
    )

    funnel_data = filtered.groupby('Funnel_Category').agg({
        'Demand': 'sum',
        'Revenue': 'sum'
    }).reset_index()

    # Kategorileri mantıklı sıraya koy (Yüksek -> Düşük)
    category_order = ['🧡 Yüksek Fiyat', '💛 Orta Fiyat', '💚 Düşük Fiyat']
    funnel_data['Order'] = funnel_data['Funnel_Category'].apply(
        lambda x: category_order.index(x) if x in category_order else 999
    )
    funnel_data = funnel_data.sort_values('Order')

    funnel_fig = go.Figure(go.Funnel(
        y=funnel_data['Funnel_Category'],
        x=funnel_data['Demand'],
        textposition="inside",
        texttemplate='<b>%{y}</b><br><b>%{x:,.0f} satış</b><br><b>(%{percentPrevious} dönüşüm)</b>',
        textfont=dict(family="Arial", size=14, color="white"),
        connector={"line": {"color": "royalblue", "dash": "dot", "width": 3}},
        marker=dict(
            color=['#E67E22', '#F1C40F', '#27AE60'],  # 3 renk: Turuncu, Sarı, Yeşil
            line=dict(width=[2, 2, 2], color=["white", "white", "white"])
        ),
        hovertemplate='<b>%{y}</b><br>Satış: %{x:,.0f}<br>Gelir: %{customdata:,.2f}₺<extra></extra>',
        customdata=funnel_data['Revenue']
    ))

    funnel_fig.update_layout(
        title={
            'text': "🎯 Fiyat Kategorisi Satış Huni Analizi",
            'font': {'size': 24, 'color': PRIMARY_COLOR},
            'x': 0.5,
            'xanchor': 'center'
        },
        font=dict(family="Arial", size=12),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor=PLOT_BG_COLOR,
        height=500,
        margin=dict(l=50, r=50, t=100, b=50)
    )

    # 3. YENİ: 3D Surface Plot - Fiyat, İndirim, Talep İlişkisi
    # Veri hazırlama: Grid oluşturmak için fiyat ve indirim aralıklarını böl
    try:
        # Fiyat ve indirim için grid değerleri oluştur
        price_min, price_max = filtered['Price'].min(), filtered['Price'].max()
        discount_min, discount_max = filtered['Discount'].min(), filtered['Discount'].max()

        # Grid boyutunu makul tut
        price_grid = np.linspace(price_min, price_max, 20)
        discount_grid = np.linspace(discount_min, discount_max, 20)

        # Her grid noktası için talep tahmini (basit interpolasyon)
        demand_surface = np.zeros((len(discount_grid), len(price_grid)))

        for i, discount_val in enumerate(discount_grid):
            for j, price_val in enumerate(price_grid):
                # Bu fiyat ve indirim kombinasyonuna yakın verileri bul
                nearby_data = filtered[
                    (abs(filtered['Price'] - price_val) <= (price_max - price_min) / 10) &
                    (abs(filtered['Discount'] - discount_val) <= (discount_max - discount_min) / 10)
                    ]

                if len(nearby_data) > 0:
                    demand_surface[i, j] = nearby_data['Demand'].mean()
                else:
                    # Yakın veri yoksa, global ortalama kullan
                    demand_surface[i, j] = filtered['Demand'].mean()

        surface_fig = go.Figure(data=[go.Surface(
            x=price_grid,
            y=discount_grid * 100,  # Yüzde olarak göster
            z=demand_surface,
            colorscale='Viridis',
            showscale=True,
            hovertemplate='<b>3D Fiyat-İndirim-Talep Analizi</b><br>' +
                          'Fiyat: %{x:.1f}₺<br>' +
                          'İndirim: %{y:.1f}%<br>' +
                          'Tahmini Talep: %{z:.0f}<br>' +
                          '<extra></extra>'
        )])

        surface_fig.update_layout(
            title={
                'text': "🎪 3D Fiyat-İndirim-Talep Yüzey Analizi",
                'font': {'size': 24, 'color': PRIMARY_COLOR},
                'x': 0.5,
                'xanchor': 'center'
            },
            scene=dict(
                xaxis_title="Fiyat (₺)",
                yaxis_title="İndirim Oranı (%)",
                zaxis_title="Talep Miktarı",
                bgcolor=PLOT_BG_COLOR,
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            font=dict(family="Arial", size=12),
            paper_bgcolor="#FFFFFF",
            height=600,
            margin=dict(l=50, r=50, t=100, b=50)
        )

    except Exception as e:
        # Hata durumunda basit bir uyarı grafiği göster
        surface_fig = go.Figure()
        surface_fig.add_annotation(
            text=f"⚠️ 3D Yüzey grafiği oluşturulamadı.<br>Veri yetersiz veya hata: {str(e)[:100]}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color=DANGER_COLOR),
            bgcolor="rgba(255,240,240,0.8)",
            bordercolor=DANGER_COLOR,
            borderwidth=2
        )
        surface_fig.update_layout(
            title={
                'text': "🎪 3D Fiyat-İndirim-Talep Yüzey Analizi",
                'font': {'size': 24, 'color': PRIMARY_COLOR},
                'x': 0.5,
                'xanchor': 'center'
            },
            height=400,
            paper_bgcolor="#FFFFFF",
            margin=dict(t=100)
        )

    # 4. Rakip Fiyat Karşılaştırması (mevcut)
    competitor_data = filtered.groupby('Product_ID').agg({
        'Price': 'mean',
        'Competitor_Price': 'mean',
        'Revenue': 'sum'
    }).reset_index()

    competitor_data = competitor_data.sort_values('Revenue', ascending=False)

    comp_fig = go.Figure()

    comp_fig.add_trace(go.Bar(
        x=[f"Ürün {x}" for x in competitor_data['Product_ID']],
        y=competitor_data['Price'],
        name='Bizim Fiyat',
        marker_color=ACCENT_COLOR,
        text=[f'{val:.1f}₺' for val in competitor_data['Price']],
        textposition='outside'
    ))

    comp_fig.add_trace(go.Bar(
        x=[f"Ürün {x}" for x in competitor_data['Product_ID']],
        y=competitor_data['Competitor_Price'],
        name='Rakip Fiyat',
        marker_color=DANGER_COLOR,
        text=[f'{val:.1f}₺' for val in competitor_data['Competitor_Price']],
        textposition='outside'
    ))

    comp_fig.update_layout(
        title={
            'text': "🆚 Tüm Ürünler için Rakip Fiyat Karşılaştırması",
            'font': {'size': 24, 'color': PRIMARY_COLOR},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(title="Ürün"),
        yaxis=dict(title="Fiyat (₺)"),
        plot_bgcolor=PLOT_BG_COLOR,
        paper_bgcolor="#FFFFFF",
        font=dict(family="Arial", size=12),
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=100)
    )

    # 5. Haftalık Fiyat Trendi (mevcut)
    filtered['Week_Date'] = filtered['Date'] - pd.to_timedelta(filtered['Date'].dt.dayofweek, unit='d')
    trend_data = filtered.groupby('Week_Date').agg({
        'Price': 'mean',
        'Competitor_Price': 'mean'
    }).reset_index()

    trend_fig = go.Figure()

    trend_fig.add_trace(go.Scatter(
        x=trend_data['Week_Date'],
        y=trend_data['Price'],
        mode='lines+markers',
        name='Bizim Ortalama Fiyat',
        line=dict(color=ACCENT_COLOR, width=3)
    ))

    trend_fig.add_trace(go.Scatter(
        x=trend_data['Week_Date'],
        y=trend_data['Competitor_Price'],
        mode='lines+markers',
        name='Rakip Ortalama Fiyat',
        line=dict(color=DANGER_COLOR, width=3, dash='dash')
    ))

    trend_fig.update_layout(
        title={
            'text': "📈 Haftalık Fiyat Trendi",
            'font': {'size': 24, 'color': PRIMARY_COLOR},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(title="Tarih"),
        yaxis=dict(title="Fiyat (₺)"),
        plot_bgcolor=PLOT_BG_COLOR,
        paper_bgcolor="#FFFFFF",
        font=dict(family="Arial", size=12),
        margin=dict(t=100)
    )

    # KPI hesaplamaları (mevcut)
    avg_price = f"{filtered['Price'].mean():.2f}₺"

    most_profitable = filtered.groupby('Product_ID')['Revenue'].sum().idxmax()
    most_profitable_text = f"Ürün {most_profitable}"

    total_revenue = f"{filtered['Revenue'].sum():,.2f}₺"

    return (
        scatter_fig,
        funnel_fig,  # YENİ: Funnel Chart
        surface_fig,  # YENİ: 3D Surface Plot
        comp_fig,
        trend_fig,
        avg_price,
        most_profitable_text,
        total_revenue
    )