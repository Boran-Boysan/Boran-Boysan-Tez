import dash
from dash import html, dcc, Input, Output
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
from datetime import timedelta
import os

dash.register_page(__name__, path="/lstm_forecast", name="LSTM Tahmin")

# Veri yükle
df = pd.read_csv("demand_forecasting_data_cleaned.csv")
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
            html.H2("🧠 LSTM Derin Öğrenme ile Akıllı Talep Tahmini", style={
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
                html.H4("🎛️ LSTM Model Parametreleri", style={
                    'color': PRIMARY_COLOR,
                    'marginBottom': '20px',
                    'textAlign': 'center',
                    'borderBottom': f'2px solid {ACCENT_COLOR}',
                    'paddingBottom': '10px'
                }),
                html.Label("🧠 Ürün Seçiniz:", style={
                    'fontWeight': 'bold',
                    'color': PRIMARY_COLOR,
                    'marginBottom': '10px',
                    'display': 'block',
                    'fontSize': '1.1em'
                }),
                dcc.Dropdown(
                    id='lstm-product-select',
                    options=[{'label': f'Ürün {i}', 'value': i} for i in sorted(df['Product_ID'].unique())],
                    placeholder="Bir ürün seçiniz...",
                    style={
                        'marginBottom': '15px',
                        'fontSize': '1rem'
                    }
                )
            ], style=filter_container_style)
        ]),

        # Main LSTM Forecast Graph
        html.Div([
            html.Div("🧠 LSTM Genel Tahmin Analizi ve Model Performansı", style=graph_header_style),
            html.Div([
                dcc.Graph(id='lstm-full-graph')
            ], style=graph_content_style)
        ], style=graph_style, className='graph-hover'),

        # 7-Day LSTM Forecast Graph
        html.Div([
            html.Div("📅 7 Günlük LSTM Detaylı Talep Tahmini", style=graph_header_style),
            html.Div([
                dcc.Graph(id='lstm-7day-graph')
            ], style=graph_content_style)
        ], style=graph_style, className='graph-hover'),

        # 30-Day LSTM Forecast Graph
        html.Div([
            html.Div("📊 30 Günlük LSTM Uzun Vadeli Talep Tahmini", style=graph_header_style),
            html.Div([
                dcc.Graph(id='lstm-30day-graph')
            ], style=graph_content_style)
        ], style=graph_style, className='graph-hover'),

        # Metrics Section
        html.Div([
            html.Div([
                html.H4("🧠 LSTM Model Performans Metrikleri", style={
                    'color': PRIMARY_COLOR,
                    'marginBottom': '20px',
                    'textAlign': 'center',
                    'borderBottom': f'2px solid {ACCENT_COLOR}',
                    'paddingBottom': '10px'
                }),
                html.Div(id="lstm-metrics-output", style={"fontSize": "1.1rem"})
            ], style=metrics_container_style)
        ])

    ], style={
        "padding": "30px",
        "backgroundColor": PLOT_BG_COLOR,
        "minHeight": "100vh"
    })


@dash.callback(
    Output('lstm-full-graph', 'figure'),
    Output('lstm-7day-graph', 'figure'),
    Output('lstm-30day-graph', 'figure'),
    Output('lstm-metrics-output', 'children'),
    Input('lstm-product-select', 'value')
)
def update_lstm_forecast(product_id):
    if not product_id:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Lütfen bir ürün seçiniz",
            plot_bgcolor=PLOT_BG_COLOR,
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12)
        )
        return empty_fig, empty_fig, empty_fig, ""

    model_path = f"models/lstm_models/{product_id}.h5"
    if not os.path.exists(model_path):
        error_fig = go.Figure()
        error_fig.update_layout(
            title=f"Model bulunamadı: {product_id}",
            plot_bgcolor=PLOT_BG_COLOR,
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12)
        )
        error_message = html.Div([
            html.Div([
                html.H5("❌ Model Hatası", style={'color': DANGER_COLOR, 'marginBottom': '15px'}),
                html.P(f"LSTM modeli bulunamadı: {model_path}", style={'fontSize': '1.1em', 'color': DANGER_COLOR}),
                html.P("Lütfen model dosyasının doğru konumda olduğundan emin olun.",
                       style={'fontSize': '1rem', 'fontStyle': 'italic'})
            ])
        ])
        return error_fig, error_fig, error_fig, error_message

    model = load_model(model_path)
    data = df[df['Product_ID'] == product_id].copy()
    data = data.groupby('Date')['Demand'].sum().reset_index()
    data['Demand'] = data['Demand'].rolling(window=7).mean()
    data.dropna(inplace=True)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data['Demand'].values.reshape(-1, 1))

    look_back = 30
    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i - look_back:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    y_true = scaler.inverse_transform(y)
    y_pred = scaler.inverse_transform(model.predict(X, verbose=0))

    last_seq = scaled[-look_back:]
    future_preds = []
    for _ in range(30):
        pred = model.predict(last_seq.reshape(1, look_back, 1), verbose=0)
        future_preds.append(pred[0, 0])
        last_seq = np.append(last_seq[1:], pred, axis=0)

    future_values = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()
    future_dates = pd.date_range(start=data['Date'].max() + timedelta(days=1), periods=30)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    avg = y_true.mean()
    rmse_pct = (rmse / avg) * 100

    # Enhanced metrics with styling
    def yorum_renk(m, good, mid):
        if m < good:
            return SUCCESS_COLOR
        elif m < mid:
            return WARNING_COLOR
        return DANGER_COLOR

    yorumlar = html.Div([
        html.Div([
            html.Span("🧠 RMSE / Ortalama Talep: ", style={'fontWeight': 'bold'}),
            html.Span(f"%{rmse_pct:.2f}", style={'fontSize': '1.2em', 'fontWeight': 'bold'}),
            html.Div(
                "✅ LSTM Mükemmel performans" if rmse_pct < 20 else
                "⚠️ LSTM Kabul edilebilir performans" if rmse_pct < 40 else
                "❌ LSTM Performans iyileştirmesi gerekli",
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
                "✅ LSTM Çok iyi doğruluk" if mape < 10 else
                "⚠️ LSTM Kabul edilebilir doğruluk" if mape < 20 else
                "❌ LSTM Doğruluk iyileştirmesi gerekli",
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
                "✅ LSTM Güçlü model uyumu" if r2 >= 0.7 else
                "⚠️ LSTM Orta seviye model uyumu" if r2 >= 0.5 else
                "❌ LSTM Zayıf model uyumu",
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
                html.H5("🧠 LSTM Temel İstatistikler", style={'color': PRIMARY_COLOR, 'marginBottom': '15px'}),
                html.P(f"📈 Ortalama Talep: {avg:,.2f} adet",
                       style={'fontSize': '1.1em', 'marginBottom': '8px'}),
                html.P(f"📊 RMSE: {rmse:,.2f}", style={'fontSize': '1.1em', 'marginBottom': '8px'}),
                html.P(f"🎯 MAPE: %{mape:.2f}", style={'fontSize': '1.1em', 'marginBottom': '8px'}),
                html.P(f"📈 R² Skoru: {r2:.4f}", style={'fontSize': '1.1em', 'marginBottom': '15px'}),
            ], style={'marginBottom': '20px'}),
            html.Hr(style={'border': f'1px solid {ACCENT_COLOR}', 'margin': '20px 0'}),
            html.H5("🎯 LSTM Performans Değerlendirmesi", style={'color': PRIMARY_COLOR, 'marginBottom': '15px'}),
            yorumlar
        ])
    ])

    history_dates = data['Date'].iloc[look_back:].reset_index(drop=True)

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
        x=history_dates,
        y=y_true.flatten(),
        mode='lines+markers',
        name='📊 Gerçek Talep',
        line=dict(color=PRIMARY_COLOR, width=2),
        marker=dict(size=6)
    ))
    fig_full.add_trace(go.Scatter(
        x=history_dates,
        y=y_pred.flatten(),
        mode='lines',
        name='🧠 LSTM Modeli (Geçmiş)',
        line=dict(color=ACCENT_COLOR, width=4, dash='solid'),
        opacity=0.9
    ))
    fig_full.add_trace(go.Scatter(
        x=future_dates,
        y=future_values,
        mode='lines+markers',
        name='🚀 LSTM Tahmin (30 Gün)',
        line=dict(color=SECONDARY_COLOR, width=3),
        marker=dict(size=8)
    ))
    fig_full = style_graph(fig_full, f"🧠 {product_id} Ürünü - LSTM ile Tahmin Analizi")
    fig_full.update_xaxes(title="📅 Tarih")
    fig_full.update_yaxes(title="📊 Talep Miktarı")

    # Enhanced 7-Day Graph
    fig_7 = go.Figure()
    fig_7.add_trace(go.Scatter(
        x=future_dates[:7],
        y=future_values[:7],
        mode='lines+markers',
        name='📅 7 Günlük LSTM Tahmin',
        line=dict(color=SUCCESS_COLOR, width=3),
        marker=dict(size=10, symbol='diamond'),
        fill='tozeroy',
        fillcolor=f'rgba(39,174,96,0.2)'
    ))
    fig_7 = style_graph(fig_7, "📅 7 Günlük LSTM Detaylı Talep Tahmini")
    fig_7.update_xaxes(title="📅 Tarih")
    fig_7.update_yaxes(title="📊 Tahmin Edilen Talep")

    # Enhanced 30-Day Graph
    fig_30 = go.Figure()
    fig_30.add_trace(go.Scatter(
        x=future_dates,
        y=future_values,
        mode='lines+markers',
        name='📊 30 Günlük LSTM Tahmin',
        line=dict(color=WARNING_COLOR, width=3),
        marker=dict(size=8),
        fill='tozeroy',
        fillcolor=f'rgba(243,156,18,0.2)'
    ))
    fig_30 = style_graph(fig_30, "📊 30 Günlük LSTM Uzun Vadeli Talep Tahmini")
    fig_30.update_xaxes(title="📅 Tarih")
    fig_30.update_yaxes(title="📊 Tahmin Edilen Talep")

    return fig_full, fig_7, fig_30, metrics


