import dash
from dash import html, dcc, Input, Output
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import base64

dash.register_page(__name__, path="/lineer_regresyon", name="Lineer Regresyon")

# Veri yükleme
df = pd.read_csv("demand_forecasting_data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['Product_ID'] = df['Product_ID'].astype('category')

# Design constants matching talep.py
PLOT_BG_COLOR = '#ECF0F1'
PRIMARY_COLOR = '#2C3E50'
ACCENT_COLOR = '#E67E22'
SECONDARY_COLOR = '#3498DB'
SUCCESS_COLOR = '#27AE60'
WARNING_COLOR = '#F39C12'
DANGER_COLOR = '#E74C3C'

# Grafik kutu stili
graph_style = {
    'boxShadow': '0 8px 32px rgba(0,0,0,0.1)',
    'borderRadius': '15px',
    'padding': '0',
    'marginBottom': '30px',
    'backgroundColor': '#fff',
    'border': '1px solid #e0e0e0',
    'overflow': 'hidden',
    'transition': 'all 0.3s ease'
}

# Header style for graph containers
graph_header_style = {
    'background': f'linear-gradient(135deg, {PRIMARY_COLOR}, #34495E)',
    'color': 'white',
    'padding': '15px 20px',
    'fontWeight': 'bold',
    'fontSize': '1.1em',
    'borderBottom': f'3px solid {ACCENT_COLOR}',
    'margin': '0'
}

# Content style for graphs
graph_content_style = {
    'padding': '20px'
}

# Filter container style
filter_container_style = {
    'background': 'white',
    'borderRadius': '12px',
    'padding': '25px',
    'marginBottom': '25px',
    'boxShadow': '0 4px 15px rgba(0,0,0,0.1)',
    'border': f'2px solid {ACCENT_COLOR}'
}

# Metrics container style
metrics_container_style = {
    'background': 'white',
    'borderRadius': '12px',
    'padding': '25px',
    'marginTop': '25px',
    'boxShadow': '0 4px 15px rgba(0,0,0,0.1)',
    'border': f'1px solid #e0e0e0'
}


def layout():
    return html.Div([
        # Main Title
        html.Div([
            html.H2("🔮 Log Dönüşümlü Lineer Regresyon ile Talep Tahmini", style={
                'textAlign': 'center',
                'color': PRIMARY_COLOR,
                'marginBottom': '30px',
                'fontWeight': 'bold',
                'textShadow': '2px 2px 4px rgba(0,0,0,0.1)',
                'fontSize': '2rem'
            })
        ]),

        # Filter Section
        html.Div([
            html.Div([
                html.H4("🎛️ Model Parametreleri", style={
                    'color': PRIMARY_COLOR,
                    'marginBottom': '20px',
                    'textAlign': 'center',
                    'borderBottom': f'2px solid {ACCENT_COLOR}',
                    'paddingBottom': '10px'
                }),
                html.Label("📦 Ürün Seçiniz:", style={
                    'fontWeight': 'bold',
                    'color': PRIMARY_COLOR,
                    'marginBottom': '10px',
                    'display': 'block',
                    'fontSize': '1.1em'
                }),
                dcc.Dropdown(
                    id='reg-log-product-select',
                    options=[{'label': f'Ürün {i}', 'value': i} for i in sorted(df['Product_ID'].unique())],
                    placeholder="Bir ürün seçiniz...",
                    style={
                        'marginBottom': '15px',
                        'fontSize': '1rem'
                    }
                ),

            ], style=filter_container_style)
        ]),

        # Main Forecast Graph
        html.Div([
            html.Div("📈 Genel Tahmin Analizi ve Model Performansı", style=graph_header_style),
            html.Div([
                dcc.Graph(id='reg-log-forecast-graph')
            ], style=graph_content_style)
        ], style=graph_style, className='graph-hover'),

        # 7-Day Forecast Graph
        html.Div([
            html.Div("📅 7 Günlük Detaylı Talep Tahmini", style=graph_header_style),
            html.Div([
                dcc.Graph(id='reg-log-7day-graph')
            ], style=graph_content_style)
        ], style=graph_style, className='graph-hover'),

        # 30-Day Forecast Graph
        html.Div([
            html.Div("📊 30 Günlük Uzun Vadeli Talep Tahmini", style=graph_header_style),
            html.Div([
                dcc.Graph(id='reg-log-30day-graph')
            ], style=graph_content_style)
        ], style=graph_style, className='graph-hover'),

        # Metrics Section
        html.Div([
            html.Div([
                html.H4("📊 Model Performans Metrikleri", style={
                    'color': PRIMARY_COLOR,
                    'marginBottom': '20px',
                    'textAlign': 'center',
                    'borderBottom': f'2px solid {ACCENT_COLOR}',
                    'paddingBottom': '10px'
                }),
                html.Div(id="reg-log-metrics", style={"fontSize": "1.1rem"})
            ], style=metrics_container_style)
        ])

    ], style={
        "padding": "30px",
        "backgroundColor": PLOT_BG_COLOR,
        "minHeight": "100vh"
    })


@dash.callback(
    Output('reg-log-forecast-graph', 'figure'),
    Output('reg-log-7day-graph', 'figure'),
    Output('reg-log-30day-graph', 'figure'),
    Output('reg-log-metrics', 'children'),
    Input('reg-log-product-select', 'value')
)
def update_log_forecast(selected_product):
    if not selected_product:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Lütfen bir ürün seçiniz",
            plot_bgcolor=PLOT_BG_COLOR,
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12)
        )
        return empty_fig, empty_fig, empty_fig, ""

    dff = df[df['Product_ID'] == selected_product]
    daily_demand = dff.groupby('Date')['Demand'].sum().reset_index()

    # Outlier filtrele
    threshold = daily_demand['Demand'].quantile(0.99)
    daily_demand = daily_demand[daily_demand['Demand'] < threshold]

    # Log dönüşüm
    daily_demand['Log_Demand'] = np.log1p(daily_demand['Demand'])
    daily_demand['Date_ordinal'] = daily_demand['Date'].map(pd.Timestamp.toordinal)

    model = LinearRegression()
    model.fit(daily_demand[['Date_ordinal']], daily_demand['Log_Demand'])

    y_pred_log = model.predict(daily_demand[['Date_ordinal']])
    y_true = daily_demand['Demand']
    y_pred = np.expm1(y_pred_log)

    # Metrikler
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    avg_demand = y_true.mean()
    rmse_pct = (rmse / avg_demand) * 100
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = model.score(daily_demand[['Date_ordinal']], daily_demand['Log_Demand'])

    # Yorumlar with enhanced styling
    def yorum_renk(m, good, mid):
        if m < good:
            return SUCCESS_COLOR
        elif m < mid:
            return WARNING_COLOR
        return DANGER_COLOR

    yorumlar = html.Div([
        html.Div([
            html.Span("📊 RMSE / Ortalama Talep: ", style={'fontWeight': 'bold'}),
            html.Span(f"%{rmse_pct:.2f}", style={'fontSize': '1.2em', 'fontWeight': 'bold'}),
            html.Div(
                "✅ Mükemmel performans" if rmse_pct < 20 else
                "⚠️ Kabul edilebilir performans" if rmse_pct < 40 else
                "❌ Performans iyileştirmesi gerekli",
                style={'fontSize': '0.9em', 'marginTop': '5px', 'fontStyle': 'italic'}
            )
        ], style={
            'backgroundColor': yorum_renk(rmse_pct, 20, 40),
            'color': 'white',
            'padding': '15px',
            'borderRadius': '10px',
            'marginBottom': '10px',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'
        }),

        html.Div([
            html.Span("🎯 MAPE (Ortalama Mutlak Yüzde Hata): ", style={'fontWeight': 'bold'}),
            html.Span(f"%{mape:.2f}", style={'fontSize': '1.2em', 'fontWeight': 'bold'}),
            html.Div(
                "✅ Çok iyi doğruluk" if mape < 10 else
                "⚠️ Kabul edilebilir doğruluk" if mape < 20 else
                "❌ Doğruluk iyileştirmesi gerekli",
                style={'fontSize': '0.9em', 'marginTop': '5px', 'fontStyle': 'italic'}
            )
        ], style={
            'backgroundColor': yorum_renk(mape, 10, 20),
            'color': 'white',
            'padding': '15px',
            'borderRadius': '10px',
            'marginBottom': '10px',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'
        }),

        html.Div([
            html.Span("📈 R² Determination Skoru: ", style={'fontWeight': 'bold'}),
            html.Span(f"{r2:.4f}", style={'fontSize': '1.2em', 'fontWeight': 'bold'}),
            html.Div(
                "✅ Güçlü model uyumu" if r2 >= 0.7 else
                "⚠️ Orta seviye model uyumu" if r2 >= 0.5 else
                "❌ Zayıf model uyumu",
                style={'fontSize': '0.9em', 'marginTop': '5px', 'fontStyle': 'italic'}
            )
        ], style={
            'backgroundColor': yorum_renk(-r2, -0.7, -0.5),
            'color': 'white',
            'padding': '15px',
            'borderRadius': '10px',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'
        }),
    ])

    metrics = html.Div([
        html.Div([
            html.Div([
                html.H5("📊 Temel İstatistikler", style={'color': PRIMARY_COLOR, 'marginBottom': '15px'}),
                html.P(f"📈 Ortalama Talep: {avg_demand:,.2f} adet", style={'fontSize': '1.1em', 'marginBottom': '8px'}),
                html.P(f"📊 RMSE: {rmse:,.2f}", style={'fontSize': '1.1em', 'marginBottom': '8px'}),
                html.P(f"🎯 MAPE: %{mape:.2f}", style={'fontSize': '1.1em', 'marginBottom': '8px'}),
                html.P(f"📈 R² Skoru: {r2:.4f}", style={'fontSize': '1.1em', 'marginBottom': '15px'}),
            ], style={'marginBottom': '20px'}),
            html.Hr(style={'border': f'1px solid {ACCENT_COLOR}', 'margin': '20px 0'}),
            html.H5("🎯 Performans Değerlendirmesi", style={'color': PRIMARY_COLOR, 'marginBottom': '15px'}),
            yorumlar
        ])
    ])

    # Tahmin tarihleri
    future_7d = pd.date_range(daily_demand['Date'].max() + pd.Timedelta(days=1), periods=7)
    future_30d = pd.date_range(daily_demand['Date'].max() + pd.Timedelta(days=1), periods=30)
    ordinal_7d = future_7d.map(pd.Timestamp.toordinal)
    ordinal_30d = future_30d.map(pd.Timestamp.toordinal)
    y_7d = np.expm1(model.predict(ordinal_7d.to_numpy().reshape(-1, 1)))
    y_30d = np.expm1(model.predict(ordinal_30d.to_numpy().reshape(-1, 1)))

    # Enhanced graph styling function
    def style_graph(fig, title):
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=18, color=PRIMARY_COLOR),
                x=0.5
            ),
            plot_bgcolor=PLOT_BG_COLOR,
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12),
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                title_font=dict(size=14, color=PRIMARY_COLOR)
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                title_font=dict(size=14, color=PRIMARY_COLOR)
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(255,255,255,0.8)"
            ),
            hovermode="x unified",
            height=450
        )
        return fig

    # Grafik 1: Gerçek + geçmiş + 30 günlük
    full_fig = go.Figure()
    full_fig.add_trace(go.Scatter(
        x=daily_demand['Date'],
        y=y_true,
        mode='lines+markers',
        name='📊 Gerçek Talep',
        line=dict(color=PRIMARY_COLOR, width=2),
        marker=dict(size=6)
    ))
    full_fig.add_trace(go.Scatter(
        x=daily_demand['Date'],
        y=y_pred,
        mode='lines',
        name='🔮 Model Tahmini (Geçmiş)',
        line=dict(color=ACCENT_COLOR, width=4, dash='solid'),
        opacity=0.9
    ))
    full_fig.add_trace(go.Scatter(
        x=future_30d,
        y=y_30d,
        mode='lines+markers',
        name='🚀 Gelecek Tahmini (30 Gün)',
        line=dict(color=SECONDARY_COLOR, width=3),
        marker=dict(size=8)
    ))
    full_fig = style_graph(full_fig, f"🔮 {selected_product} Ürünü - Regresyon ile Tahmin Analizi")
    full_fig.update_xaxes(title="📅 Tarih")
    full_fig.update_yaxes(title="📊 Talep Miktarı")

    # Grafik 2: 7 günlük tahmin
    fig_7d = go.Figure()
    fig_7d.add_trace(go.Scatter(
        x=future_7d,
        y=y_7d,
        mode='lines+markers',
        name='📅 7 Günlük Tahmin',
        line=dict(color=SUCCESS_COLOR, width=3),
        marker=dict(size=10, symbol='diamond'),
        fill='tozeroy',
        fillcolor=f'rgba(39,174,96,0.2)'
    ))
    fig_7d = style_graph(fig_7d, "📅 7 Günlük Detaylı Talep Tahmini")
    fig_7d.update_xaxes(title="📅 Tarih")
    fig_7d.update_yaxes(title="📊 Tahmin Edilen Talep")

    # Grafik 3: 30 günlük tahmin
    fig_30d = go.Figure()
    fig_30d.add_trace(go.Scatter(
        x=future_30d,
        y=y_30d,
        mode='lines+markers',
        name='📊 30 Günlük Tahmin',
        line=dict(color=WARNING_COLOR, width=3),
        marker=dict(size=8),
        fill='tozeroy',
        fillcolor=f'rgba(243,156,18,0.2)'
    ))
    fig_30d = style_graph(fig_30d, "📊 30 Günlük Uzun Vadeli Talep Tahmini")
    fig_30d.update_xaxes(title="📅 Tarih")
    fig_30d.update_yaxes(title="📊 Tahmin Edilen Talep")

    return full_fig, fig_7d, fig_30d, metrics
