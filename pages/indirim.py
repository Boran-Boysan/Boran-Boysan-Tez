import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import register_page, html, dcc, Input, Output, State, callback
import dash.exceptions
import numpy as np

register_page(__name__, path="/kampanya", name="Kampanya ve İndirim Analizi")

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

# Kampanya ve indirim metrikleri
df['Revenue'] = df['Price'] * df['Demand']
df['Discount_Amount'] = df['Price'] * df['Discount']
df['Discounted_Price'] = df['Price'] * (1 - df['Discount'])
df['Discount_Category'] = pd.cut(df['Discount'],
                                 bins=[0, 0.05, 0.15, 0.30, 1.0],
                                 labels=['💚 Düşük (0-5%)', '💛 Orta (5-15%)', '🧡 Yüksek (15-30%)',
                                         '🔴 Çok Yüksek (30%+)'],
                                 include_lowest=True)

# Kampanya etkisi metrikleri
df['Campaign_ROI'] = (df['Revenue'] - df['Discount_Amount']) / (df['Discount_Amount'] + 1)
df['Campaign_Effectiveness'] = df['Marketing_Effect'] * df['Demand']

PLOT_BG_COLOR = '#ECF0F1'
PRIMARY_COLOR = '#2C3E50'
ACCENT_COLOR = '#E67E22'
SUCCESS_COLOR = '#27AE60'
WARNING_COLOR = '#F39C12'
DANGER_COLOR = '#E74C3C'


def layout():
    return html.Div([
        html.H3("🎯 Kampanya ve İndirim Performans Analizi", style={
            'textAlign': 'center',
            'color': PRIMARY_COLOR,
            'marginBottom': '30px',
            'fontSize': '2rem',
            'fontWeight': 'bold'
        }),

        # KPI kartları
        html.Div([
            html.Div([
                html.H4("📊 Ortalama İndirim Oranı", style={'textAlign': 'center', 'margin': '0'}),
                html.H2(id="avg-discount-rate", style={'textAlign': 'center', 'color': '#fff', 'margin': '10px 0'})
            ], className='kpi-card', style={
                'background': f'linear-gradient(135deg, {ACCENT_COLOR}, #D35400)',
                'padding': '20px',
                'borderRadius': '10px',
                'color': 'white',
                'width': '22%',
                'margin': '10px'
            }),

            html.Div([
                html.H4("💰 Toplam İndirim Tutarı", style={'textAlign': 'center', 'margin': '0'}),
                html.H2(id="total-discount-amount", style={'textAlign': 'center', 'color': '#fff', 'margin': '10px 0'})
            ], className='kpi-card', style={
                'background': f'linear-gradient(135deg, {SUCCESS_COLOR}, #229954)',
                'padding': '20px',
                'borderRadius': '10px',
                'color': 'white',
                'width': '22%',
                'margin': '10px'
            }),

            html.Div([
                html.H4("🚀 En Etkili Kampanya", style={'textAlign': 'center', 'margin': '0'}),
                html.H2(id="best-campaign", style={'textAlign': 'center', 'color': '#fff', 'margin': '10px 0'})
            ], className='kpi-card', style={
                'background': f'linear-gradient(135deg, {DANGER_COLOR}, #C0392B)',
                'padding': '20px',
                'borderRadius': '10px',
                'color': 'white',
                'width': '22%',
                'margin': '10px'
            }),

            html.Div([
                html.H4("📈 Kampanya ROI", style={'textAlign': 'center', 'margin': '0'}),
                html.H2(id="avg-campaign-roi", style={'textAlign': 'center', 'color': '#fff', 'margin': '10px 0'})
            ], className='kpi-card', style={
                'background': f'linear-gradient(135deg, #3498DB, #2980B9)',
                'padding': '20px',
                'borderRadius': '10px',
                'color': 'white',
                'width': '22%',
                'margin': '10px'
            })
        ], style={'display': 'flex', 'justifyContent': 'center', 'flexWrap': 'wrap', 'marginBottom': '30px'}),

        # Grafik bölümleri - app.py formatı ile uyumlu
        html.Div([
            html.Div("🎯 Kampanya Türlerine Göre Performans Analizi", className='graph-header'),
            html.Div([dcc.Graph(id='campaign-performance-graph')], className='graph-content')
        ], className='graph-container'),

        html.Div([
            html.Div("📐 Kampanya Etkililik Radar Analizi", className='graph-header'),
            html.Div([dcc.Graph(id='campaign-radar-chart')], className='graph-content')
        ], className='graph-container'),

        html.Div([
            html.Div("🎭 Kampanya → İndirim Stratejisi Akış Analizi", className='graph-header'),
            html.Div([dcc.Graph(id='campaign-sankey-diagram')], className='graph-content')
        ], className='graph-container'),

        html.Div([
            html.Div("📅 Haftalık Kampanya Performans Trendi", className='graph-header'),
            html.Div([dcc.Graph(id='campaign-timeline-graph')], className='graph-content')
        ], className='graph-container'),

        html.Div([
            html.Div("💸 İndirim Oranı vs Talep Korelasyonu", className='graph-header'),
            html.Div([dcc.Graph(id='discount-demand-correlation')], className='graph-content')
        ], className='graph-container'),

        html.Div([
            html.Div("📈 Kampanya ROI (Yatırım Getirisi) Analizi", className='graph-header'),
            html.Div([dcc.Graph(id='campaign-roi-graph')], className='graph-content')
        ], className='graph-container'),

        html.Div([
            html.Div("🌍 Mevsimsel Kampanya Performansı", className='graph-header'),
            html.Div([dcc.Graph(id='seasonal-campaign-graph')], className='graph-content')
        ], className='graph-container'),
    ])


@callback(
    Output("campaign-performance-graph", "figure"),
    Output("campaign-radar-chart", "figure"),  # YENİ: Radar Chart output
    Output("campaign-sankey-diagram", "figure"),  # YENİ: Sankey Diagram output
    Output("campaign-timeline-graph", "figure"),
    Output("discount-demand-correlation", "figure"),
    Output("campaign-roi-graph", "figure"),
    Output("seasonal-campaign-graph", "figure"),
    Output("avg-discount-rate", "children"),
    Output("total-discount-amount", "children"),
    Output("best-campaign", "children"),
    Output("avg-campaign-roi", "children"),
    Input("apply-button", "n_clicks"),
    State("product-dropdown", "value"),
    State("marketing-dropdown", "value"),
    State("price-slider", "value"),
    State("discount-slider", "value"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
    State("url", "pathname")
)
def update_campaign_graphs(n, product_id, marketing, price_range, discount_range, start_date, end_date, pathname):
    if pathname != "/kampanya":
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

    # 1. Kampanya Performans Analizi
    campaign_performance = filtered.groupby('Marketing_Campaign').agg({
        'Demand': 'sum',
        'Revenue': 'sum',
        'Marketing_Effect': 'mean',
        'Discount': 'mean'
    }).reset_index().sort_values('Revenue', ascending=True)

    campaign_perf_fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('📊 Kampanya Türüne Göre Gelir', '🎯 Kampanya Türüne Göre Talep'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )

    # Gelir grafiği
    campaign_perf_fig.add_trace(
        go.Bar(
            x=campaign_performance['Revenue'],
            y=campaign_performance['Marketing_Campaign'],
            name='Gelir',
            orientation='h',
            marker_color=ACCENT_COLOR,
            text=[f'{val:,.0f}₺' for val in campaign_performance['Revenue']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Gelir: %{x:,.2f}₺<extra></extra>'
        ),
        row=1, col=1
    )

    # Talep grafiği
    campaign_perf_fig.add_trace(
        go.Bar(
            x=campaign_performance['Demand'],
            y=campaign_performance['Marketing_Campaign'],
            name='Talep',
            orientation='h',
            marker_color=SUCCESS_COLOR,
            text=[f'{val:,.0f}' for val in campaign_performance['Demand']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Talep: %{x:,.0f}<extra></extra>'
        ),
        row=1, col=2
    )

    campaign_perf_fig.update_layout(
        title={
            'text': "🎯 Kampanya Türlerine Göre Performans Analizi",
            'font': {'size': 24, 'color': PRIMARY_COLOR},
            'x': 0.5,
            'xanchor': 'center'
        },
        plot_bgcolor=PLOT_BG_COLOR,
        paper_bgcolor="#FFFFFF",
        font=dict(family="Arial", size=12),
        height=500,
        showlegend=False,
        margin=dict(t=100)
    )

    # Alt başlıkları büyüt
    campaign_perf_fig.update_annotations(font_size=18)

    # 2. YENİ: Radar Chart - Kampanya Etkililik Skorları
    try:
        radar_data = filtered.groupby('Marketing_Campaign').agg({
            'Demand': 'sum',
            'Revenue': 'sum',
            'Marketing_Effect': 'mean',
            'Campaign_ROI': 'mean',
            'Discount': 'mean'
        }).reset_index()

        if len(radar_data) > 0:
            # Skorları 0-100 arasına normalize et
            def normalize_score(series):
                if series.max() == series.min():
                    return pd.Series([50] * len(series), index=series.index)
                return ((series - series.min()) / (series.max() - series.min()) * 100).fillna(50)

            radar_data['Talep_Score'] = normalize_score(radar_data['Demand'])
            radar_data['Gelir_Score'] = normalize_score(radar_data['Revenue'])
            radar_data['Etki_Score'] = normalize_score(radar_data['Marketing_Effect'])
            radar_data['ROI_Score'] = normalize_score(radar_data['Campaign_ROI'])
            radar_data['Indirim_Score'] = 100 - normalize_score(radar_data['Discount'])

            categories = ['📈 Talep', '💰 Gelir', '🎯 Pazarlama Etkisi', '📊 ROI', '💸 İndirim Etkinliği']

            radar_fig = go.Figure()

            colors = [ACCENT_COLOR, SUCCESS_COLOR, DANGER_COLOR, '#3498DB', '#9B59B6']

            for i, campaign in enumerate(radar_data['Marketing_Campaign']):
                scores = [
                    radar_data.loc[radar_data['Marketing_Campaign'] == campaign, 'Talep_Score'].iloc[0],
                    radar_data.loc[radar_data['Marketing_Campaign'] == campaign, 'Gelir_Score'].iloc[0],
                    radar_data.loc[radar_data['Marketing_Campaign'] == campaign, 'Etki_Score'].iloc[0],
                    radar_data.loc[radar_data['Marketing_Campaign'] == campaign, 'ROI_Score'].iloc[0],
                    radar_data.loc[radar_data['Marketing_Campaign'] == campaign, 'Indirim_Score'].iloc[0]
                ]

                # Renk kodunu güvenli şekilde çevir
                color = colors[i % len(colors)]
                if color.startswith('#'):
                    r = int(color[1:3], 16)
                    g = int(color[3:5], 16)
                    b = int(color[5:7], 16)
                else:
                    r, g, b = 100, 100, 100  # Varsayılan renk

                radar_fig.add_trace(go.Scatterpolar(
                    r=scores,
                    theta=categories,
                    fill='toself',
                    name=campaign,
                    line=dict(color=color, width=3),
                    marker=dict(size=8),
                    fillcolor=f'rgba({r}, {g}, {b}, 0.2)',
                    hovertemplate=f'<b>{campaign}</b><br>%{{theta}}: %{{r:.1f}}/100<extra></extra>'
                ))

            radar_fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100],
                        tickvals=[20, 40, 60, 80, 100],
                        ticktext=['20', '40', '60', '80', '100'],
                        gridcolor='rgba(0,0,0,0.3)',
                        linecolor='rgba(0,0,0,0.3)'
                    ),
                    bgcolor=PLOT_BG_COLOR,
                    angularaxis=dict(
                        tickfont=dict(size=12, color=PRIMARY_COLOR)
                    )
                ),
                title={
                    'text': "📐 Kampanya Etkililik Radar Analizi",
                    'font': {'size': 24, 'color': PRIMARY_COLOR},
                    'x': 0.5,
                    'xanchor': 'center'
                },
                font=dict(family="Arial", size=12),
                paper_bgcolor="#FFFFFF",
                height=600,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.05
                ),
                margin=dict(t=100)
            )
        else:
            raise ValueError("Radar için veri yok")

    except Exception as e:
        # Hata durumunda basit bir grafik göster
        radar_fig = go.Figure()
        radar_fig.add_annotation(
            text=f"⚠️ Radar grafiği oluşturulamadı.<br>Veri yetersiz.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color=DANGER_COLOR),
            bgcolor="rgba(255,240,240,0.8)",
            bordercolor=DANGER_COLOR,
            borderwidth=2
        )
        radar_fig.update_layout(
            title={
                'text': "📐 Kampanya Etkililik Radar Analizi",
                'font': {'size': 24, 'color': PRIMARY_COLOR},
                'x': 0.5,
                'xanchor': 'center'
            },
            height=400,
            paper_bgcolor="#FFFFFF",
            margin=dict(t=100)
        )

    # 3. YENİ: Sankey Diagram - Kampanya Akışı Analizi
    try:
        # Kampanya Türü → İndirim Seviyesi → Satış Sonucu akışı

        # İndirim seviyelerini basitleştir - Güvenli versiyonu
        if len(filtered) > 0:
            discount_quantiles = filtered['Discount'].quantile([0.33, 0.67]).values
            filtered_copy = filtered.copy()
            filtered_copy['Simple_Discount'] = pd.cut(
                filtered_copy['Discount'],
                bins=[0, discount_quantiles[0], discount_quantiles[1], 1.0],
                labels=['Düşük İndirim', 'Orta İndirim', 'Yüksek İndirim'],
                include_lowest=True
            )

            # Satış performansı kategorisi
            demand_quantiles = filtered_copy['Demand'].quantile([0.33, 0.67]).values
            filtered_copy['Sales_Performance'] = pd.cut(
                filtered_copy['Demand'],
                bins=[0, demand_quantiles[0], demand_quantiles[1], float('inf')],
                labels=['Düşük Satış', 'Orta Satış', 'Yüksek Satış'],
                include_lowest=True
            )

            # NaN değerleri temizle
            filtered_clean = filtered_copy.dropna(subset=['Marketing_Campaign', 'Simple_Discount', 'Sales_Performance'])

            if len(filtered_clean) > 0:
                # Kampanya -> İndirim akışı
                campaign_discount = filtered_clean.groupby(
                    ['Marketing_Campaign', 'Simple_Discount']).size().reset_index(name='Count')

                # Node listesi oluştur
                campaigns = list(filtered_clean['Marketing_Campaign'].unique())
                discounts = ['Düşük İndirim', 'Orta İndirim', 'Yüksek İndirim']

                all_nodes = campaigns + discounts
                node_colors = [ACCENT_COLOR] * len(campaigns) + [SUCCESS_COLOR] * len(discounts)

                # Link listesi oluştur
                sources = []
                targets = []
                values = []

                for _, row in campaign_discount.iterrows():
                    if row['Marketing_Campaign'] in campaigns and row['Simple_Discount'] in discounts:
                        source_idx = campaigns.index(row['Marketing_Campaign'])
                        target_idx = len(campaigns) + discounts.index(row['Simple_Discount'])
                        sources.append(source_idx)
                        targets.append(target_idx)
                        values.append(row['Count'])

                sankey_fig = go.Figure(data=[go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=all_nodes,
                        color=node_colors
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values,
                        color='rgba(100,100,100,0.4)'
                    )
                )])

                sankey_fig.update_layout(
                    title={
                        'text': "🎭 Kampanya → İndirim Stratejisi Akış Analizi",
                        'font': {'size': 24, 'color': PRIMARY_COLOR},
                        'x': 0.5,
                        'xanchor': 'center'
                    },
                    font=dict(family="Arial", size=12),
                    paper_bgcolor="#FFFFFF",
                    height=500,
                    margin=dict(t=100)
                )
            else:
                raise ValueError("Temizlenmiş veri boş")
        else:
            raise ValueError("Filtrelenmiş veri boş")

    except Exception as e:
        # Hata durumunda basit bir alternatif grafik göster
        sankey_fig = go.Figure()
        sankey_fig.add_annotation(
            text=f"⚠️ Sankey diyagramı oluşturulamadı.<br>Veri yetersiz veya hata oluştu.<br>Lütfen filtrelerinizi kontrol edin.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color=DANGER_COLOR),
            bgcolor="rgba(255,240,240,0.8)",
            bordercolor=DANGER_COLOR,
            borderwidth=2,
            borderpad=10
        )
        sankey_fig.update_layout(
            title={
                'text': "🎭 Kampanya → İndirim Stratejisi Akış Analizi",
                'font': {'size': 24, 'color': PRIMARY_COLOR},
                'x': 0.5,
                'xanchor': 'center'
            },
            height=400,
            paper_bgcolor="#FFFFFF",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(t=100)
        )

    # 4. Kampanya Zaman Serisi - Temizlenmiş versiyon
    filtered['Week_Date'] = filtered['Date'] - pd.to_timedelta(filtered['Date'].dt.dayofweek, unit='d')
    campaign_timeline = filtered.groupby(['Week_Date', 'Marketing_Campaign']).agg({
        'Demand': 'sum',
        'Revenue': 'sum'
    }).reset_index()

    timeline_fig = go.Figure()

    campaigns = campaign_timeline['Marketing_Campaign'].unique()
    colors_timeline = [ACCENT_COLOR, SUCCESS_COLOR, DANGER_COLOR, '#3498DB', '#9B59B6']
    line_styles = ['solid', 'dash', 'dot', 'dashdot', 'longdash']

    for i, campaign in enumerate(campaigns):
        campaign_data = campaign_timeline[campaign_timeline['Marketing_Campaign'] == campaign]
        timeline_fig.add_trace(go.Scatter(
            x=campaign_data['Week_Date'],
            y=campaign_data['Demand'],
            mode='lines+markers',
            name=campaign,
            line=dict(
                color=colors_timeline[i % len(colors_timeline)],
                width=3,
                dash=line_styles[i % len(line_styles)]
            ),
            marker=dict(size=8, symbol=['circle', 'diamond', 'square', 'triangle-up', 'star'][i % 5]),
            connectgaps=False,
            hovertemplate=f'<b>{campaign}</b><br>Hafta: %{{x|%d.%m.%Y}}<br>Talep: %{{y:,.0f}}<extra></extra>'
        ))

    timeline_fig.update_layout(
        title={
            'text': "📅 Haftalık Kampanya Performans Trendi",
            'font': {'size': 24, 'color': PRIMARY_COLOR},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(title="Hafta", tickformat='%d.%m'),
        yaxis=dict(title="Haftalık Toplam Talep"),
        plot_bgcolor=PLOT_BG_COLOR,
        paper_bgcolor="#FFFFFF",
        font=dict(family="Arial", size=12),
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        margin=dict(r=150, t=100)
    )

    # 5. İndirim-Talep Korelasyonu
    correlation_data = filtered.sample(min(1000, len(filtered)))

    discount_corr_fig = px.scatter(
        correlation_data,
        x='Discount',
        y='Demand',
        color='Marketing_Campaign',
        size='Price',
        title="💸 İndirim Oranı vs Talep Korelasyonu",
        labels={
            'Discount': 'İndirim Oranı',
            'Demand': 'Talep',
            'Price': 'Fiyat'
        },
        hover_data=['Product_ID', 'Revenue']
    )

    # Trend çizgisi
    discount_corr_fig.add_trace(
        px.scatter(correlation_data, x='Discount', y='Demand', trendline="ols").data[1]
    )

    discount_corr_fig.update_layout(
        title={
            'text': "💸 İndirim Oranı vs Talep Korelasyonu",
            'font': {'size': 24, 'color': PRIMARY_COLOR},
            'x': 0.5,
            'xanchor': 'center'
        },
        plot_bgcolor=PLOT_BG_COLOR,
        paper_bgcolor="#FFFFFF",
        font=dict(family="Arial", size=12),
        margin=dict(t=100)
    )

    discount_corr_fig.update_xaxes(tickformat='.0%')

    # 6. Kampanya ROI Analizi
    roi_data = filtered.groupby('Marketing_Campaign').agg({
        'Campaign_ROI': 'mean',
        'Revenue': 'sum',
        'Discount_Amount': 'sum'
    }).reset_index()

    roi_data['Net_Profit'] = roi_data['Revenue'] - roi_data['Discount_Amount']
    roi_data = roi_data.sort_values('Campaign_ROI', ascending=True)

    roi_fig = go.Figure()

    roi_fig.add_trace(go.Bar(
        x=roi_data['Campaign_ROI'],
        y=roi_data['Marketing_Campaign'],
        orientation='h',
        name='Kampanya ROI',
        marker=dict(
            color=roi_data['Campaign_ROI'],
            colorscale='RdYlGn',
            colorbar=dict(title="ROI Değeri")
        ),
        text=[f'{val:.2f}' for val in roi_data['Campaign_ROI']],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>ROI: %{x:.2f}<br>Net Kar: %{customdata:,.2f}₺<extra></extra>',
        customdata=roi_data['Net_Profit']
    ))

    roi_fig.update_layout(
        title={
            'text': "📈 Kampanya ROI (Yatırım Getirisi) Analizi",
            'font': {'size': 24, 'color': PRIMARY_COLOR},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(title="ROI Değeri"),
        yaxis=dict(title="Kampanya Türü"),
        plot_bgcolor=PLOT_BG_COLOR,
        paper_bgcolor="#FFFFFF",
        font=dict(family="Arial", size=12),
        margin=dict(t=100)
    )

    # 7. Mevsimsel Kampanya Analizi
    seasonal_campaign = filtered.groupby(['Seasonal_Trend', 'Marketing_Campaign']).agg({
        'Demand': 'sum',
        'Revenue': 'sum'
    }).reset_index()

    seasonal_fig = px.sunburst(
        seasonal_campaign,
        path=['Seasonal_Trend', 'Marketing_Campaign'],
        values='Revenue',
        title="🌍 Mevsimsel Kampanya Performansı (Gelir Bazlı)",
        color='Revenue',
        color_continuous_scale='Viridis'
    )

    seasonal_fig.update_layout(
        title={
            'text': "🌍 Mevsimsel Kampanya Performansı (Gelir Bazlı)",
            'font': {'size': 24, 'color': PRIMARY_COLOR},
            'x': 0.5,
            'xanchor': 'center'
        },
        font=dict(family="Arial", size=12),
        paper_bgcolor="#FFFFFF",
        margin=dict(t=100)
    )

    # KPI hesaplamaları - Güvenli versiyonlar
    try:
        avg_discount = f"{filtered['Discount'].mean() * 100:.1f}%" if len(filtered) > 0 else "0.0%"
        total_discount = f"{filtered['Discount_Amount'].sum():,.2f}₺" if len(filtered) > 0 else "0.00₺"

        # En etkili kampanya - Güvenli hesaplama
        if len(filtered) > 0:
            campaign_effectiveness = filtered.groupby('Marketing_Campaign')['Campaign_Effectiveness'].sum()
            if len(campaign_effectiveness) > 0:
                best_campaign_name = campaign_effectiveness.idxmax()
            else:
                best_campaign_name = "Veri Yok"
        else:
            best_campaign_name = "Veri Yok"

        avg_roi = f"{filtered['Campaign_ROI'].mean():.2f}" if len(filtered) > 0 else "0.00"

    except Exception as e:
        # Hata durumunda varsayılan değerler
        avg_discount = "0.0%"
        total_discount = "0.00₺"
        best_campaign_name = "Hata"
        avg_roi = "0.00"

    return (
        campaign_perf_fig,
        radar_fig,  # YENİ: Radar Chart
        sankey_fig,  # YENİ: Sankey Diagram
        timeline_fig,
        discount_corr_fig,
        roi_fig,
        seasonal_fig,
        avg_discount,
        total_discount,
        best_campaign_name,
        avg_roi
    )
