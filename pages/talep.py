import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import register_page, html, dcc, Input, Output, State, callback
import dash.exceptions
import numpy as np

register_page(__name__, path="/talep", name="Talep için Görselleştirme")

# Veri Yükleme
df = pd.read_csv("demand_forecasting_data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df[df["Demand"].notna() & (df["Demand"] > 0)].copy()
df["Year"] = df["Date"].dt.year
df["Week"] = df["Date"].dt.isocalendar().week
df["Month"] = df["Date"].dt.month
df["Month_Name"] = df["Date"].dt.strftime('%B')
df["Weekday"] = df["Date"].dt.day_name()
df["Quarter"] = df["Date"].dt.quarter
df["Public_Holiday"] = df["Public_Holiday"].astype(int)
df["Seasonal_Trend"] = df["Seasonal_Trend"].astype("category")
df["Marketing_Campaign"] = df["Marketing_Campaign"].astype("category")
df["Product_ID"] = df["Product_ID"].astype("category")

# Ana tema renkleri (app.py ile uyumlu)
PRIMARY_COLOR = '#2C3E50'
ACCENT_COLOR = '#E67E22'
SECONDARY_COLOR = '#3498DB'
SUCCESS_COLOR = '#27AE60'
WARNING_COLOR = '#F39C12'
DANGER_COLOR = '#E74C3C'
BACKGROUND_COLOR = '#ECF0F1'
TEXT_COLOR = '#FFFFFF'
PLOT_BG_COLOR = '#FAFAFA'
GRAPH_COLORS = ['#E67E22', '#3498DB', '#27AE60', '#9B59B6', '#F39C12', '#E74C3C', '#1ABC9C', '#34495E']


def layout():
    return html.Div([
        # Sayfa başlığı
        html.Div([
            html.H2("📈 Gelişmiş Talep Analizi ve Görselleştirme", style={
                'textAlign': 'center',
                'color': PRIMARY_COLOR,
                'marginBottom': '30px',
                'fontWeight': 'bold',
                'textShadow': '2px 2px 4px rgba(0,0,0,0.1)'
            })
        ]),

        # KPI Kartları - Talep sayfasına özel (5 KPI)
        html.Div([
            html.Div([
                html.Div([
                    html.I(className="fas fa-chart-line", style={'fontSize': '2rem', 'marginBottom': '10px'}),
                    html.H3("Maksimum Talep", style={'margin': '0', 'fontSize': '1.2rem'}),
                    html.H1(id="max-demand-kpi", style={'margin': '10px 0', 'fontSize': '2.5rem', 'fontWeight': 'bold'})
                ], style={'textAlign': 'center'})
            ], className='kpi-card', style={
                "background": f"linear-gradient(135deg, {DANGER_COLOR}, #C0392B)",
                "padding": "25px",
                "color": "white",
                "borderRadius": "15px",
                "width": "18%",
                "minHeight": "140px",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center"
            }),

            html.Div([
                html.Div([
                    html.I(className="fas fa-chart-line-down", style={'fontSize': '2rem', 'marginBottom': '10px'}),
                    html.H3("Minimum Talep", style={'margin': '0', 'fontSize': '1.2rem'}),
                    html.H1(id="min-demand-kpi", style={'margin': '10px 0', 'fontSize': '2.5rem', 'fontWeight': 'bold'})
                ], style={'textAlign': 'center'})
            ], className='kpi-card', style={
                "background": f"linear-gradient(135deg, {SECONDARY_COLOR}, #2980B9)",
                "padding": "25px",
                "color": "white",
                "borderRadius": "15px",
                "width": "18%",
                "minHeight": "140px",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center"
            }),

            html.Div([
                html.Div([
                    html.I(className="fas fa-calculator", style={'fontSize': '2rem', 'marginBottom': '10px'}),
                    html.H3("Toplam Talep", style={'margin': '0', 'fontSize': '1.2rem'}),
                    html.H1(id="total-demand-kpi",
                            style={'margin': '10px 0', 'fontSize': '2.5rem', 'fontWeight': 'bold'})
                ], style={'textAlign': 'center'})
            ], className='kpi-card', style={
                "background": f"linear-gradient(135deg, {SUCCESS_COLOR}, #229954)",
                "padding": "25px",
                "color": "white",
                "borderRadius": "15px",
                "width": "18%",
                "minHeight": "140px",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center"
            }),

            html.Div([
                html.Div([
                    html.I(className="fas fa-calendar-week", style={'fontSize': '2rem', 'marginBottom': '10px'}),
                    html.H3("En Yoğun Hafta", style={'margin': '0', 'fontSize': '1.2rem'}),
                    html.H1(id="peak-week-kpi", style={'margin': '10px 0', 'fontSize': '2.5rem', 'fontWeight': 'bold'})
                ], style={'textAlign': 'center'})
            ], className='kpi-card', style={
                "background": f"linear-gradient(135deg, {WARNING_COLOR}, #D68910)",
                "padding": "25px",
                "color": "white",
                "borderRadius": "15px",
                "width": "18%",
                "minHeight": "140px",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center"
            }),

            html.Div([
                html.Div([
                    html.I(className="fas fa-trending-up", style={'fontSize': '2rem', 'marginBottom': '10px'}),
                    html.H3("Talep Volatilitesi", style={'margin': '0', 'fontSize': '1.2rem'}),
                    html.H1(id="volatility-kpi", style={'margin': '10px 0', 'fontSize': '2.5rem', 'fontWeight': 'bold'})
                ], style={'textAlign': 'center'})
            ], className='kpi-card', style={
                "background": f"linear-gradient(135deg, #9B59B6, #8E44AD)",
                "padding": "25px",
                "color": "white",
                "borderRadius": "15px",
                "width": "18%",
                "minHeight": "140px",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center"
            })
        ], style={
            "display": "flex",
            "gap": "15px",
            "flexWrap": "wrap",
            "justifyContent": "center",
            "marginBottom": "40px"
        }),

        # Grafik konteynerları - app.py stili ile uyumlu
        html.Div([
            html.Div("⏰ Zaman Serisine Göre Talep Analizi", className='graph-header'),
            html.Div([dcc.Graph(id="main-timeseries-graph")], className='graph-content')
        ], className='graph-container'),

        html.Div([
            html.Div("🌟 Mevsimsel ve Yıllık Trend Analizi", className='graph-header'),
            html.Div([dcc.Graph(id="seasonal-yearly-combined")], className='graph-content')
        ], className='graph-container'),

        html.Div([
            html.Div("🏷️ Ürün Performans Analizi", className='graph-header'),
            html.Div([dcc.Graph(id="product-analysis")], className='graph-content')
        ], className='graph-container'),

        html.Div([
            html.Div("📢 Kampanya Etkinlik Analizi", className='graph-header'),
            html.Div([dcc.Graph(id="campaign-effectiveness")], className='graph-content')
        ], className='graph-container'),

        html.Div([
            html.Div("📊 Haftalık Trend ve Hareketli Ortalama", className='graph-header'),
            html.Div([dcc.Graph(id="weekly-analysis")], className='graph-content')
        ], className='graph-container'),

        html.Div([
            html.Div("🔥 Talep Yoğunluk Haritası", className='graph-header'),
            html.Div([dcc.Graph(id="demand-heatmap")], className='graph-content')
        ], className='graph-container'),
    ])


@callback(
    Output("main-timeseries-graph", "figure"),
    Output("seasonal-yearly-combined", "figure"),
    Output("product-analysis", "figure"),
    Output("campaign-effectiveness", "figure"),
    Output("weekly-analysis", "figure"),
    Output("demand-heatmap", "figure"),
    Output("max-demand-kpi", "children"),
    Output("min-demand-kpi", "children"),
    Output("total-demand-kpi", "children"),
    Output("peak-week-kpi", "children"),
    Output("volatility-kpi", "children"),
    Input("apply-button", "n_clicks"),
    State("product-dropdown", "value"),
    State("marketing-dropdown", "value"),
    State("price-slider", "value"),
    State("discount-slider", "value"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
    State("url", "pathname")
)
def update_demand_analysis(n, product_id, marketing, price_range, discount_range, start_date, end_date, pathname):
    if pathname != "/talep":
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

    # 1. Gelişmiş Zaman Serisi Grafiği
    daily_demand = filtered.groupby('Date')['Demand'].sum().reset_index()

    timeseries_fig = go.Figure()

    # Ana talep çizgisi
    timeseries_fig.add_trace(go.Scatter(
        x=daily_demand["Date"],
        y=daily_demand["Demand"],
        mode="lines+markers",
        name="Günlük Talep",
        line=dict(color=ACCENT_COLOR, width=3),
        marker=dict(size=6, color=ACCENT_COLOR),
        fill='tonexty',
        fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(ACCENT_COLOR)) + [0.2])}",
        hovertemplate="<b>Tarih:</b> %{x|%d.%m.%Y}<br><b>Talep:</b> %{y:,.0f} adet<extra></extra>"
    ))

    # 7 günlük hareketli ortalama
    daily_demand['MA7'] = daily_demand['Demand'].rolling(window=7, min_periods=1).mean()
    timeseries_fig.add_trace(go.Scatter(
        x=daily_demand["Date"],
        y=daily_demand["MA7"],
        mode="lines",
        name="7 Günlük Ortalama",
        line=dict(color=SECONDARY_COLOR, width=2, dash="dash"),
        hovertemplate="<b>Tarih:</b> %{x|%d.%m.%Y}<br><b>7G Ort:</b> %{y:,.0f} adet<extra></extra>"
    ))

    timeseries_fig.update_layout(
        title={
            'text': "⏰ Zaman Serisine Göre Günlük Talep Analizi",
            'font': {'size': 24, 'color': PRIMARY_COLOR},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(
            title="Tarih",
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            title_font=dict(size=14, color=PRIMARY_COLOR)
        ),
        yaxis=dict(
            title="Talep Miktarı",
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            title_font=dict(size=14, color=PRIMARY_COLOR)
        ),
        plot_bgcolor=PLOT_BG_COLOR,
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=12),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        height=500,
        margin=dict(t=100)
    )

    # 2. Birleşik Mevsimsel ve Yıllık Analiz
    seasonal_yearly_fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("🌟 Mevsimsel Trend Dağılımı", "📅 Yıllık Talep Karşılaştırması"),
        specs=[[{"type": "xy"}, {"type": "xy"}]]
    )

    # Mevsimsel box plot
    for i, season in enumerate(filtered['Seasonal_Trend'].unique()):
        season_data = filtered[filtered['Seasonal_Trend'] == season]['Demand']
        seasonal_yearly_fig.add_trace(
            go.Box(
                y=season_data,
                name=season,
                marker_color=GRAPH_COLORS[i % len(GRAPH_COLORS)],
                boxpoints='outliers',
                jitter=0.3,
                pointpos=-1.8
            ),
            row=1, col=1
        )

    # Yıllık bar chart
    yearly_data = filtered.groupby('Year')['Demand'].sum().reset_index()
    seasonal_yearly_fig.add_trace(
        go.Bar(
            x=yearly_data['Year'],
            y=yearly_data['Demand'],
            name='Yıllık Toplam',
            marker_color=ACCENT_COLOR,
            text=yearly_data['Demand'],
            texttemplate='%{text:,.0f}',
            textposition='outside'
        ),
        row=1, col=2
    )

    seasonal_yearly_fig.update_layout(
        title={
            'text': "🌟 Mevsimsel ve Yıllık Trend Analizi",
            'font': {'size': 24, 'color': PRIMARY_COLOR},
            'x': 0.5,
            'xanchor': 'center'
        },
        plot_bgcolor=PLOT_BG_COLOR,
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=12),
        height=500,
        showlegend=False,
        margin=dict(t=100)
    )

    # Alt başlıkları büyüt
    seasonal_yearly_fig.update_annotations(font_size=18)

    # 3. Ürün Performans Analizi (5 ürün için) - SİYAH YAZI
    product_data = filtered.groupby('Product_ID').agg({
        'Demand': 'sum',
        'Price': 'mean'
    }).reset_index()
    product_data['Revenue'] = product_data['Demand'] * product_data['Price']

    # 5 ürün olduğunu varsayarak hepsini göster
    product_data = product_data.sort_values('Demand', ascending=False)

    product_fig = go.Figure(go.Treemap(
        labels=[f"Ürün {pid}" for pid in product_data['Product_ID']],
        values=product_data['Demand'],
        parents=[""] * len(product_data),
        text=[f"Ürün {pid}<br>Talep: {demand:,.0f}<br>Gelir: {revenue:,.0f}₺"
              for pid, demand, revenue in zip(product_data['Product_ID'],
                                              product_data['Demand'],
                                              product_data['Revenue'])],
        textinfo="text",
        textfont=dict(size=14, color='black'),  # SİYAH YAZI
        marker=dict(
            colorscale='Viridis',
            colorbar=dict(title="Talep Miktarı")
        ),
        hovertemplate='<b>%{label}</b><br>Talep: %{value:,.0f}<extra></extra>'
    ))

    product_fig.update_layout(
        title={
            'text': "🏷️ Tüm Ürünlerin Performans Haritası",
            'font': {'size': 24, 'color': PRIMARY_COLOR},
            'x': 0.5,
            'xanchor': 'center'
        },
        font=dict(family="Arial, sans-serif", size=12),
        height=500,
        margin=dict(t=100)
    )

    # 4. Kampanya Etkinlik Analizi (Pie Chart) - SİYAH YAZI
    campaign_data = filtered.groupby('Marketing_Campaign')['Demand'].sum().reset_index()
    campaign_data = campaign_data.sort_values('Demand', ascending=False)

    # Renk paleti
    colors = GRAPH_COLORS[:len(campaign_data)]

    campaign_fig = go.Figure(data=[go.Pie(
        labels=campaign_data['Marketing_Campaign'],
        values=campaign_data['Demand'],
        hole=0.4,  # Donut chart için
        marker=dict(
            colors=colors,
            line=dict(color='white', width=3)
        ),
        textinfo='label+percent+value',
        texttemplate='<b>%{label}</b><br>%{percent}<br>%{value:,.0f} adet',
        textfont=dict(size=12, color='black'),  # SİYAH YAZI
        hovertemplate='<b>%{label}</b><br>Talep: %{value:,.0f}<br>Oran: %{percent}<extra></extra>',
        pull=[0.1 if i == 0 else 0.05 for i in range(len(campaign_data))]  # En büyük dilimi öne çıkar
    )])

    campaign_fig.update_layout(
        title={
            'text': "📢 Kampanya Türlerine Göre Talep Dağılımı",
            'font': {'size': 24, 'color': PRIMARY_COLOR},
            'x': 0.5,
            'xanchor': 'center'
        },
        font=dict(family="Arial, sans-serif", size=12),
        height=500,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        ),
        annotations=[
            dict(
                text="Toplam<br>Talep",
                x=0.5, y=0.5,
                font=dict(size=16, color='black'),  # Merkez yazı da siyah
                showarrow=False
            )
        ],
        margin=dict(t=100)
    )

    # 5. Haftalık Analiz
    weekly_data = filtered.groupby("Week").agg({
        'Demand': ['sum', 'mean', 'std']
    }).round(2)
    weekly_data.columns = ['Toplam', 'Ortalama', 'Std_Sapma']
    weekly_data = weekly_data.reset_index()
    weekly_data['Variation_Coeff'] = (weekly_data['Std_Sapma'] / weekly_data['Ortalama'] * 100).fillna(0)

    weekly_fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("📈 Haftalık Toplam Talep", "📊 Haftalık Değişkenlik Katsayısı (%)"),
        vertical_spacing=0.1
    )

    # Haftalık toplam talep
    weekly_fig.add_trace(
        go.Scatter(
            x=weekly_data["Week"],
            y=weekly_data["Toplam"],
            mode="lines+markers",
            name="Haftalık Toplam",
            line=dict(color=ACCENT_COLOR, width=3),
            fill='tozeroy',
            fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(ACCENT_COLOR)) + [0.3])}"
        ),
        row=1, col=1
    )

    # Değişkenlik katsayısı
    weekly_fig.add_trace(
        go.Bar(
            x=weekly_data["Week"],
            y=weekly_data["Variation_Coeff"],
            name="Değişkenlik %",
            marker_color=SECONDARY_COLOR,
            text=weekly_data["Variation_Coeff"],
            texttemplate='%{text:.1f}%',
            textposition='outside'
        ),
        row=2, col=1
    )

    weekly_fig.update_layout(
        title={
            'text': "📊 Haftalık Trend ve Hareketli Ortalama",
            'font': {'size': 24, 'color': PRIMARY_COLOR},
            'x': 0.5,
            'xanchor': 'center'
        },
        plot_bgcolor=PLOT_BG_COLOR,
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=12),
        height=600,
        showlegend=False,
        margin=dict(t=100)
    )

    # Alt başlıkları büyüt
    weekly_fig.update_annotations(font_size=18)

    # 6. Talep Yoğunluk Haritası (Heatmap)
    # Hafta vs Ürün ID heatmap
    pivot_data = filtered.groupby(['Week', 'Product_ID'])['Demand'].sum().reset_index()
    heatmap_pivot = pivot_data.pivot(index='Product_ID', columns='Week', values='Demand').fillna(0)

    heatmap_fig = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns,
        y=[f"Ürün {pid}" for pid in heatmap_pivot.index],
        colorscale='YlOrRd',
        hovertemplate='<b>Hafta:</b> %{x}<br><b>Ürün:</b> %{y}<br><b>Talep:</b> %{z:,.0f}<extra></extra>',
        colorbar=dict(title="Talep Miktarı")
    ))

    heatmap_fig.update_layout(
        title={
            'text': "🔥 Hafta-Ürün Talep Yoğunluk Haritası",
            'font': {'size': 24, 'color': PRIMARY_COLOR},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(title="Hafta", title_font=dict(size=14, color=PRIMARY_COLOR)),
        yaxis=dict(title="Ürün ID", title_font=dict(size=14, color=PRIMARY_COLOR)),
        font=dict(family="Arial, sans-serif", size=12),
        height=500,
        margin=dict(t=100)
    )

    # KPI hesaplamaları
    max_demand_val = f"{filtered['Demand'].max():,}"
    min_demand_val = f"{filtered['Demand'].min():,}"
    total_demand_val = f"{filtered['Demand'].sum():,}"

    # En yoğun hafta
    weekly_demand = filtered.groupby('Week')['Demand'].sum()
    peak_week = weekly_demand.idxmax()
    peak_week_val = f"Hafta {peak_week}"

    # Volatilite (Değişkenlik katsayısı)
    volatility = (filtered['Demand'].std() / filtered['Demand'].mean() * 100)
    volatility_val = f"{volatility:.1f}%"

    return (
        timeseries_fig,
        seasonal_yearly_fig,
        product_fig,
        campaign_fig,
        weekly_fig,
        heatmap_fig,
        max_demand_val,
        min_demand_val,
        total_demand_val,
        peak_week_val,
        volatility_val
    )
