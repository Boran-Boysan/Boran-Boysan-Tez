import dash
from dash import html, dcc, Input, Output
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import plotly.graph_objects as go

dash.register_page(__name__, path="/polinom-ridge", name="Polinom Regresyon Tahmini")

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
            html.H2("📐 Polinom Ridge Regresyon ile Optimize Talep Tahmini", style={
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
                html.H4("🎛️ Polinom Ridge Model Parametreleri", style={
                    'color': PRIMARY_COLOR,
                    'marginBottom': '20px',
                    'textAlign': 'center',
                    'borderBottom': f'2px solid {ACCENT_COLOR}',
                    'paddingBottom': '10px'
                }),
                html.Label("📐 Ürün Seçiniz:", style={
                    'fontWeight': 'bold',
                    'color': PRIMARY_COLOR,
                    'marginBottom': '10px',
                    'display': 'block',
                    'fontSize': '1.1em'
                }),
                dcc.Dropdown(
                    id='ridge-product-select',
                    options=[{'label': f'Ürün {i}', 'value': i} for i in sorted(df['Product_ID'].unique())],
                    placeholder="Bir ürün seçiniz...",
                    style={
                        'marginBottom': '15px',
                        'fontSize': '1rem'
                    }
                )
            ], style=filter_container_style)
        ]),

        # Main Ridge Forecast Graph
        html.Div([
            html.Div("📐 Polinom Ridge Genel Tahmin Analizi ve Model Performansı", style=graph_header_style),
            html.Div([
                dcc.Graph(id='ridge-forecast-graph')
            ], style=graph_content_style)
        ], style=graph_style, className='graph-hover'),

        # 7-Day Ridge Forecast Graph
        html.Div([
            html.Div("📅 7 Günlük Polinom Ridge Detaylı Talep Tahmini", style=graph_header_style),
            html.Div([
                dcc.Graph(id='ridge-7day-graph')
            ], style=graph_content_style)
        ], style=graph_style, className='graph-hover'),

        # 30-Day Ridge Forecast Graph
        html.Div([
            html.Div("📊 30 Günlük Polinom Ridge Uzun Vadeli Talep Tahmini", style=graph_header_style),
            html.Div([
                dcc.Graph(id='ridge-30day-graph')
            ], style=graph_content_style)
        ], style=graph_style, className='graph-hover'),

        # Metrics Section
        html.Div([
            html.Div([
                html.H4("📐 Polinom Ridge Model Performans Metrikleri", style={
                    'color': PRIMARY_COLOR,
                    'marginBottom': '20px',
                    'textAlign': 'center',
                    'borderBottom': f'2px solid {ACCENT_COLOR}',
                    'paddingBottom': '10px'
                }),
                html.Div(id="ridge-regression-metrics", style={"fontSize": "1.1rem"})
            ], style=metrics_container_style)
        ])

    ], style={
        "padding": "30px",
        "backgroundColor": PLOT_BG_COLOR,
        "minHeight": "100vh"
    })


@dash.callback(
    Output('ridge-forecast-graph', 'figure'),
    Output('ridge-7day-graph', 'figure'),
    Output('ridge-30day-graph', 'figure'),
    Output('ridge-regression-metrics', 'children'),
    Input('ridge-product-select', 'value')
)
def update_ridge_polynomial_forecast(selected_product):
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
    daily['Date_ordinal'] = daily['Date'].map(pd.Timestamp.toordinal)
    daily['Smoothed_Demand'] = daily['Demand'].ewm(span=5, adjust=False).mean()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(daily[['Date_ordinal']])
    poly = PolynomialFeatures(degree=4)
    X_poly = poly.fit_transform(X_scaled)

    # GridSearch benzeri alpha seçimi
    best_alpha, best_rmse_pct = None, float("inf")
    for alpha in [0.1, 1.0, 5.0, 10.0, 25.0, 50.0]:
        temp_model = Ridge(alpha=alpha)
        temp_model.fit(X_poly, daily['Smoothed_Demand'])
        y_temp_pred = temp_model.predict(X_poly)
        rmse_pct_temp = np.sqrt(mean_squared_error(daily['Smoothed_Demand'], y_temp_pred)) / daily[
            'Smoothed_Demand'].mean() * 100
        if rmse_pct_temp < best_rmse_pct:
            best_rmse_pct = rmse_pct_temp
            best_alpha = alpha

    model = Ridge(alpha=best_alpha)
    model.fit(X_poly, daily['Smoothed_Demand'])
    y_pred_train = model.predict(X_poly)

    # Performans metrikleri
    rmse = np.sqrt(mean_squared_error(daily['Smoothed_Demand'], y_pred_train))
    r2 = r2_score(daily['Smoothed_Demand'], y_pred_train)
    mape = mean_absolute_percentage_error(daily['Smoothed_Demand'], y_pred_train) * 100
    avg_demand = daily['Smoothed_Demand'].mean()
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
            html.Span("📐 RMSE / Ortalama Talep: ", style={'fontWeight': 'bold'}),
            html.Span(f"%{rmse_pct:.2f}", style={'fontSize': '1.2em', 'fontWeight': 'bold'}),
            html.Div(
                "✅ Ridge Mükemmel performans" if rmse_pct < 20 else
                "⚠️ Ridge Kabul edilebilir performans" if rmse_pct < 40 else
                "❌ Ridge Performans iyileştirmesi gerekli",
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
                "✅ Ridge Çok iyi doğruluk" if mape < 10 else
                "⚠️ Ridge Kabul edilebilir doğruluk" if mape < 20 else
                "❌ Ridge Doğruluk iyileştirmesi gerekli",
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
                "✅ Ridge Güçlü model uyumu" if r2 >= 0.7 else
                "⚠️ Ridge Orta seviye model uyumu" if r2 >= 0.5 else
                "❌ Ridge Zayıf model uyumu",
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

    metrics_text = html.Div([
        html.Div([
            html.Div([
                html.H5("📐 Polinom Ridge Temel İstatistikler", style={'color': PRIMARY_COLOR, 'marginBottom': '15px'}),
                html.P(f"🎯 Optimize Alpha: {best_alpha}",
                       style={'fontSize': '1.1em', 'marginBottom': '8px', 'fontWeight': 'bold', 'color': ACCENT_COLOR}),
                html.P(f"📊 Polinom Derecesi: 4", style={'fontSize': '1.1em', 'marginBottom': '8px'}),
                html.P(f"📈 Ortalama Talep: {avg_demand:,.2f} adet",
                       style={'fontSize': '1.1em', 'marginBottom': '8px'}),
                html.P(f"📊 RMSE: {rmse:,.2f}", style={'fontSize': '1.1em', 'marginBottom': '8px'}),
                html.P(f"🎯 MAPE: %{mape:.2f}", style={'fontSize': '1.1em', 'marginBottom': '8px'}),
                html.P(f"📈 R² Skoru: {r2:.4f}", style={'fontSize': '1.1em', 'marginBottom': '15px'}),
            ], style={'marginBottom': '20px'}),
            html.Hr(style={'border': f'1px solid {ACCENT_COLOR}', 'margin': '20px 0'}),
            html.H5("🎯 Ridge Performans Değerlendirmesi", style={'color': PRIMARY_COLOR, 'marginBottom': '15px'}),
            yorumlar
        ])
    ])

    # Gelecek tahminleri
    future_dates = pd.date_range(daily['Date'].max() + pd.Timedelta(days=1), periods=30)
    future_ordinals = future_dates.map(pd.Timestamp.toordinal)
    X_future_scaled = scaler.transform(future_ordinals.to_numpy().reshape(-1, 1))
    X_future_poly = poly.transform(X_future_scaled)
    y_future = model.predict(X_future_poly)

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
        y=daily['Demand'],
        mode='lines+markers',
        name='📊 Gerçek Talep',
        line=dict(color=PRIMARY_COLOR, width=2),
        marker=dict(size=6)
    ))
    fig_full.add_trace(go.Scatter(
        x=daily['Date'],
        y=y_pred_train,
        mode='lines',
        name='📐 Ridge Modeli (Geçmiş)',
        line=dict(color=ACCENT_COLOR, width=4, dash='solid'),
        opacity=0.9
    ))
    fig_full.add_trace(go.Scatter(
        x=future_dates,
        y=y_future,
        mode='lines+markers',
        name='🚀 Ridge Tahmin (30 Gün)',
        line=dict(color=SECONDARY_COLOR, width=3),
        marker=dict(size=8)
    ))
    fig_full = style_graph(fig_full, f"📐 {selected_product} Ürünü - Polinom Ridge ile Tahmin Analizi")
    fig_full.update_xaxes(title="📅 Tarih")
    fig_full.update_yaxes(title="📊 Talep Miktarı")

    # Enhanced 7-Day Graph
    fig_7d = go.Figure()
    fig_7d.add_trace(go.Scatter(
        x=future_dates[:7],
        y=y_future[:7],
        mode='lines+markers',
        name='📅 7 Günlük Ridge Tahmin',
        line=dict(color=SUCCESS_COLOR, width=3),
        marker=dict(size=10, symbol='diamond'),
        fill='tozeroy',
        fillcolor=f'rgba(39,174,96,0.2)'
    ))
    fig_7d = style_graph(fig_7d, "📅 7 Günlük Polinom Ridge Detaylı Talep Tahmini")
    fig_7d.update_xaxes(title="📅 Tarih")
    fig_7d.update_yaxes(title="📊 Tahmin Edilen Talep")

    # Enhanced 30-Day Graph
    fig_30d = go.Figure()
    fig_30d.add_trace(go.Scatter(
        x=future_dates,
        y=y_future,
        mode='lines+markers',
        name='📊 30 Günlük Ridge Tahmin',
        line=dict(color=WARNING_COLOR, width=3),
        marker=dict(size=8),
        fill='tozeroy',
        fillcolor=f'rgba(243,156,18,0.2)'
    ))
    fig_30d = style_graph(fig_30d, "📊 30 Günlük Polinom Ridge Uzun Vadeli Talep Tahmini")
    fig_30d.update_xaxes(title="📅 Tarih")
    fig_30d.update_yaxes(title="📊 Tahmin Edilen Talep")

    return fig_full, fig_7d, fig_30d, metrics_text


