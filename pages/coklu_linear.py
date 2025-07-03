import dash
from dash import html, dcc, Input, Output
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
from datetime import timedelta

dash.register_page(__name__, path="/coklu_lineer_recursive", name="Çoklu Lineer Regresyon (Recursive)")

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
            html.H2("🔄 Çoklu Özellikli Recursive Regresyon ile Akıllı Talep Tahmini", style={
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
                html.H4("🎛️ Çoklu Recursive Model Parametreleri", style={
                    'color': PRIMARY_COLOR,
                    'marginBottom': '20px',
                    'textAlign': 'center',
                    'borderBottom': f'2px solid {ACCENT_COLOR}',
                    'paddingBottom': '10px'
                }),
                html.Label("🔄 Ürün Seçiniz:", style={
                    'fontWeight': 'bold',
                    'color': PRIMARY_COLOR,
                    'marginBottom': '10px',
                    'display': 'block',
                    'fontSize': '1.1em'
                }),
                dcc.Dropdown(
                    id='coklu-reg-recursive-product-select',
                    options=[{'label': f'Ürün {i}', 'value': i} for i in sorted(df['Product_ID'].unique())],
                    placeholder="Bir ürün seçiniz...",
                    style={
                        'marginBottom': '15px',
                        'fontSize': '1rem'
                    }
                ),
                html.Div([
                    html.P("📊 Özellik Mühendisliği:",
                           style={'fontWeight': 'bold', 'color': PRIMARY_COLOR, 'marginBottom': '5px'}),
                    html.Ul([
                        html.Li("📅 Tarih Ordinali & Hafta İçi/Sonu",
                                style={'fontSize': '0.9em', 'marginBottom': '3px'}),
                        html.Li("🔄 Lag 1 & Lag 7 (Gecikmeli Değerler)",
                                style={'fontSize': '0.9em', 'marginBottom': '3px'}),
                        html.Li("📈 7-Günlük Hareketli Ortalama", style={'fontSize': '0.9em', 'marginBottom': '3px'}),
                        html.Li("🧮 Log Dönüşüm & Smoothing", style={'fontSize': '0.9em', 'marginBottom': '3px'})
                    ], style={'paddingLeft': '20px', 'color': '#5D6D7E'})
                ], style={
                    'background': f'rgba({PRIMARY_COLOR[1:]}, 0.05)',
                    'padding': '15px',
                    'borderRadius': '8px',
                    'marginTop': '10px',
                    'border': f'1px solid rgba({PRIMARY_COLOR[1:]}, 0.2)'
                })
            ], style=filter_container_style)
        ]),

        # Main Recursive Forecast Graph
        html.Div([
            html.Div("🔄 Çoklu Recursive Genel Tahmin Analizi ve Model Performansı", style=graph_header_style),
            html.Div([
                dcc.Graph(id='coklu-rec-forecast-graph')
            ], style=graph_content_style)
        ], style=graph_style, className='graph-hover'),

        # 7-Day Recursive Forecast Graph
        html.Div([
            html.Div("📅 7 Günlük Çoklu Recursive Detaylı Talep Tahmini", style=graph_header_style),
            html.Div([
                dcc.Graph(id='coklu-rec-7day-graph')
            ], style=graph_content_style)
        ], style=graph_style, className='graph-hover'),

        # 30-Day Recursive Forecast Graph
        html.Div([
            html.Div("📊 30 Günlük Çoklu Recursive Uzun Vadeli Talep Tahmini", style=graph_header_style),
            html.Div([
                dcc.Graph(id='coklu-rec-30day-graph')
            ], style=graph_content_style)
        ], style=graph_style, className='graph-hover'),

        # Metrics Section
        html.Div([
            html.Div([
                html.H4("🔄 Çoklu Recursive Model Performans Metrikleri", style={
                    'color': PRIMARY_COLOR,
                    'marginBottom': '20px',
                    'textAlign': 'center',
                    'borderBottom': f'2px solid {ACCENT_COLOR}',
                    'paddingBottom': '10px'
                }),
                html.Div(id="coklu-reg-recursive-metrics", style={"fontSize": "1.1rem"})
            ], style=metrics_container_style)
        ])

    ], style={
        "padding": "30px",
        "backgroundColor": PLOT_BG_COLOR,
        "minHeight": "100vh"
    })


@dash.callback(
    Output('coklu-rec-forecast-graph', 'figure'),
    Output('coklu-rec-7day-graph', 'figure'),
    Output('coklu-rec-30day-graph', 'figure'),
    Output('coklu-reg-recursive-metrics', 'children'),
    Input('coklu-reg-recursive-product-select', 'value')
)
def update_recursive_forecast(selected_product):
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
    daily = dff.groupby('Date')['Demand'].sum().reset_index()

    # Outlier + Smoothing
    threshold = daily['Demand'].quantile(0.95)
    daily = daily[daily['Demand'] < threshold].copy()
    daily['Smoothed_Demand'] = daily['Demand'].rolling(window=7).mean()
    daily.dropna(inplace=True)

    # Feature engineering
    daily['DayOfWeek'] = daily['Date'].dt.dayofweek
    daily['Month'] = daily['Date'].dt.month
    daily['IsWeekend'] = (daily['DayOfWeek'] >= 5).astype(int)
    daily['Lag_1'] = daily['Smoothed_Demand'].shift(1)
    daily['Lag_7'] = daily['Smoothed_Demand'].shift(7)
    daily['RollingMean_7'] = daily['Smoothed_Demand'].rolling(7).mean()
    daily.dropna(inplace=True)

    daily['Log_Smoothed_Demand'] = np.log1p(daily['Smoothed_Demand'])
    daily['Date_ordinal'] = daily['Date'].map(pd.Timestamp.toordinal)

    X = daily[['Date_ordinal', 'DayOfWeek', 'Month', 'IsWeekend', 'Lag_1', 'Lag_7', 'RollingMean_7']]
    y = daily['Log_Smoothed_Demand']

    model = LinearRegression()
    model.fit(X, y)

    y_pred_log = model.predict(X)
    y_pred = np.expm1(y_pred_log)
    y_true = daily['Smoothed_Demand']

    # Recursive prediction
    history = daily.copy()
    future_predictions = []
    future_dates = pd.date_range(start=history['Date'].max() + timedelta(days=1), periods=30)

    for date in future_dates:
        dayofweek = date.dayofweek
        month = date.month
        is_weekend = int(dayofweek >= 5)
        date_ordinal = date.toordinal()

        recent_values = history.tail(7)['Smoothed_Demand'].tolist()
        lag_1 = recent_values[-1]
        lag_7 = recent_values[0] if len(recent_values) >= 7 else lag_1
        rolling_7 = np.mean(recent_values)

        features = np.array([[date_ordinal, dayofweek, month, is_weekend, lag_1, lag_7, rolling_7]])
        pred_log = model.predict(features)[0]
        pred = np.expm1(pred_log)

        future_predictions.append((date, pred))
        history = pd.concat([history, pd.DataFrame({'Date': [date], 'Smoothed_Demand': [pred]})], ignore_index=True)

    future_result = pd.DataFrame(future_predictions, columns=["Date", "Forecasted_Smoothed_Demand"])

    # Metrikler
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y, y_pred_log)
    avg_demand = y_true.mean()
    rmse_pct = (rmse / avg_demand) * 100

    # Enhanced metrics with styling
    def yorum_renk(m, good, mid):
        if m < good:
            return SUCCESS_COLOR
        elif m < mid:
            return WARNING_COLOR
        return DANGER_COLOR

    yorumlar = html.Div([
        html.Div([
            html.Span("🔄 RMSE / Ortalama Talep: ", style={'fontWeight': 'bold'}),
            html.Span(f"%{rmse_pct:.2f}", style={'fontSize': '1.2em', 'fontWeight': 'bold'}),
            html.Div(
                "✅ Recursive Mükemmel performans" if rmse_pct < 20 else
                "⚠️ Recursive Kabul edilebilir performans" if rmse_pct < 40 else
                "❌ Recursive Performans iyileştirmesi gerekli",
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
                "✅ Recursive Çok iyi doğruluk" if mape < 10 else
                "⚠️ Recursive Kabul edilebilir doğruluk" if mape < 20 else
                "❌ Recursive Doğruluk iyileştirmesi gerekli",
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
                "✅ Recursive Güçlü model uyumu" if r2 >= 0.7 else
                "⚠️ Recursive Orta seviye model uyumu" if r2 >= 0.5 else
                "❌ Recursive Zayıf model uyumu",
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
                html.H5("🔄 Çoklu Recursive Temel İstatistikler",
                        style={'color': PRIMARY_COLOR, 'marginBottom': '15px'}),
                html.P(f"📊 Özellik Sayısı: 7 (Tarih, Hafta, Ay, Lag1, Lag7, Rolling7, Weekend)",
                       style={'fontSize': '1.1em', 'marginBottom': '8px'}),
                html.P(f"🧮 Veri İşleme: 7-günlük smoothing + Log dönüşüm",
                       style={'fontSize': '1.1em', 'marginBottom': '8px'}),
                html.P(f"📈 Ortalama Talep (7-gün smoothed): {avg_demand:,.2f} adet",
                       style={'fontSize': '1.1em', 'marginBottom': '8px'}),
                html.P(f"📊 RMSE: {rmse:,.2f}", style={'fontSize': '1.1em', 'marginBottom': '8px'}),
                html.P(f"🎯 MAPE: %{mape:.2f}", style={'fontSize': '1.1em', 'marginBottom': '8px'}),
                html.P(f"📈 R² Skoru (log): {r2:.4f}", style={'fontSize': '1.1em', 'marginBottom': '15px'}),
            ], style={'marginBottom': '20px'}),
            html.Hr(style={'border': f'1px solid {ACCENT_COLOR}', 'margin': '20px 0'}),
            html.H5("🎯 Recursive Performans Değerlendirmesi", style={'color': PRIMARY_COLOR, 'marginBottom': '15px'}),
            yorumlar
        ])
    ])

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

    # Enhanced Full Graph
    fig_full = go.Figure()
    fig_full.add_trace(go.Scatter(
        x=daily['Date'],
        y=y_true,
        mode='lines+markers',
        name='📊 Gerçek Talep',
        line=dict(color=PRIMARY_COLOR, width=2),
        marker=dict(size=6)
    ))
    fig_full.add_trace(go.Scatter(
        x=daily['Date'],
        y=y_pred,
        mode='lines',
        name='🔄 Recursive Modeli (Geçmiş)',
        line=dict(color=ACCENT_COLOR, width=4, dash='solid'),
        opacity=0.9
    ))
    fig_full.add_trace(go.Scatter(
        x=future_result['Date'],
        y=future_result['Forecasted_Smoothed_Demand'],
        mode='lines+markers',
        name='🚀 Recursive Tahmin (30 Gün)',
        line=dict(color=SECONDARY_COLOR, width=3),
        marker=dict(size=8)
    ))
    fig_full = style_graph(fig_full, f"🔄 {selected_product} Ürünü - Çoklu Recursive ile Tahmin Analizi")
    fig_full.update_xaxes(title="📅 Tarih")
    fig_full.update_yaxes(title="📊 Talep Miktarı")

    # Enhanced 7-Day Graph
    fig_7d = go.Figure()
    fig_7d.add_trace(go.Scatter(
        x=future_result['Date'][:7],
        y=future_result['Forecasted_Smoothed_Demand'][:7],
        mode='lines+markers',
        name='📅 7 Günlük Recursive Tahmin',
        line=dict(color=SUCCESS_COLOR, width=3),
        marker=dict(size=10, symbol='diamond'),
        fill='tozeroy',
        fillcolor=f'rgba(39,174,96,0.2)'
    ))
    fig_7d = style_graph(fig_7d, "📅 7 Günlük Çoklu Recursive Detaylı Talep Tahmini")
    fig_7d.update_xaxes(title="📅 Tarih")
    fig_7d.update_yaxes(title="📊 Tahmin Edilen Talep")

    # Enhanced 30-Day Graph
    fig_30d = go.Figure()
    fig_30d.add_trace(go.Scatter(
        x=future_result['Date'],
        y=future_result['Forecasted_Smoothed_Demand'],
        mode='lines+markers',
        name='📊 30 Günlük Recursive Tahmin',
        line=dict(color=WARNING_COLOR, width=3),
        marker=dict(size=8),
        fill='tozeroy',
        fillcolor=f'rgba(243,156,18,0.2)'
    ))
    fig_30d = style_graph(fig_30d, "📊 30 Günlük Çoklu Recursive Uzun Vadeli Talep Tahmini")
    fig_30d.update_xaxes(title="📅 Tarih")
    fig_30d.update_yaxes(title="📊 Tahmin Edilen Talep")

    return fig_full, fig_7d, fig_30d, metrics




