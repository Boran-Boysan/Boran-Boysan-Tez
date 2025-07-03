import dash
from dash import html, dcc, Input, Output
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

dash.register_page(__name__, path="/arima_forecast", name="ARIMA Zaman Serisi Tahmini")

df = pd.read_csv("demand_forecasting_data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['Product_ID'] = df['Product_ID'].astype('category')

# Design constants
PLOT_BG_COLOR = '#ECF0F1'
PRIMARY_COLOR = '#2C3E50'
ACCENT_COLOR = '#E67E22'
SECONDARY_COLOR = '#3498DB'
SUCCESS_COLOR = '#27AE60'
WARNING_COLOR = '#F39C12'
DANGER_COLOR = '#E74C3C'

# Grafik stilleri
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

graph_header_style = {
    'background': f'linear-gradient(135deg, {PRIMARY_COLOR}, #34495E)',
    'color': 'white',
    'padding': '15px 20px',
    'fontWeight': 'bold',
    'fontSize': '1.1em',
    'borderBottom': f'3px solid {ACCENT_COLOR}',
    'margin': '0'
}

graph_content_style = {'padding': '20px'}

filter_container_style = {
    'background': 'white',
    'borderRadius': '12px',
    'padding': '25px',
    'marginBottom': '25px',
    'boxShadow': '0 4px 15px rgba(0,0,0,0.1)',
    'border': f'2px solid {ACCENT_COLOR}'
}

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
            html.H2("📈 ARIMA ile Klasik Zaman Serisi Tahmini", style={
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
                html.H4("🎛️ ARIMA Model Parametreleri", style={
                    'color': PRIMARY_COLOR,
                    'marginBottom': '20px',
                    'textAlign': 'center',
                    'borderBottom': f'2px solid {ACCENT_COLOR}',
                    'paddingBottom': '10px'
                }),
                html.Label("📈 Ürün Seçiniz:", style={
                    'fontWeight': 'bold',
                    'color': PRIMARY_COLOR,
                    'marginBottom': '10px',
                    'display': 'block',
                    'fontSize': '1.1em'
                }),
                dcc.Dropdown(
                    id='arima-product-select',
                    options=[{'label': f'Ürün {i}', 'value': i} for i in sorted(df['Product_ID'].unique())],
                    placeholder="Bir ürün seçiniz...",
                    style={
                        'marginBottom': '15px',
                        'fontSize': '1rem'
                    }
                ),
                html.Div([
                    html.P("📈 ARIMA Özellikler:",
                           style={'fontWeight': 'bold', 'color': PRIMARY_COLOR, 'marginBottom': '5px'}),
                    html.Ul([
                        html.Li("📊 Auto ARIMA ile otomatik parametre seçimi",
                                style={'fontSize': '0.9em', 'marginBottom': '3px'}),
                        html.Li("🔍 Durağanlık testi (ADF Test)", style={'fontSize': '0.9em', 'marginBottom': '3px'}),
                        html.Li("📉 Trend ve mevsimsel ayrıştırma", style={'fontSize': '0.9em', 'marginBottom': '3px'}),
                        html.Li("⚡ AR, MA ve I bileşenleri analizi", style={'fontSize': '0.9em', 'marginBottom': '3px'})
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

        # Stationarity Test Results
        html.Div([
            html.Div("🔍 Durağanlık Testi ve Veri Hazırlık Analizi", style=graph_header_style),
            html.Div([
                html.Div(id='stationarity-results')
            ], style=graph_content_style)
        ], style=graph_style, className='graph-hover'),

        # Main ARIMA Forecast Graph
        html.Div([
            html.Div("📈 ARIMA Genel Tahmin Analizi ve Model Performansı", style=graph_header_style),
            html.Div([
                dcc.Graph(id='arima-full-graph')
            ], style=graph_content_style)
        ], style=graph_style, className='graph-hover'),

        # Seasonal Decomposition
        html.Div([
            html.Div("🔄 Zaman Serisi Ayrıştırma Analizi (Trend, Mevsimsel, Residual)", style=graph_header_style),
            html.Div([
                dcc.Graph(id='arima-decomposition-graph')
            ], style=graph_content_style)
        ], style=graph_style, className='graph-hover'),

        # 7-Day ARIMA Forecast Graph
        html.Div([
            html.Div("📅 7 Günlük ARIMA Detaylı Talep Tahmini", style=graph_header_style),
            html.Div([
                dcc.Graph(id='arima-7day-graph')
            ], style=graph_content_style)
        ], style=graph_style, className='graph-hover'),

        # 30-Day ARIMA Forecast Graph
        html.Div([
            html.Div("📊 30 Günlük ARIMA Uzun Vadeli Talep Tahmini", style=graph_header_style),
            html.Div([
                dcc.Graph(id='arima-30day-graph')
            ], style=graph_content_style)
        ], style=graph_style, className='graph-hover'),

        # Metrics Section
        html.Div([
            html.Div([
                html.H4("📈 ARIMA Model Performans Metrikleri", style={
                    'color': PRIMARY_COLOR,
                    'marginBottom': '20px',
                    'textAlign': 'center',
                    'borderBottom': f'2px solid {ACCENT_COLOR}',
                    'paddingBottom': '10px'
                }),
                html.Div(id="arima-metrics-output", style={"fontSize": "1.1rem"})
            ], style=metrics_container_style)
        ])

    ], style={
        "padding": "30px",
        "backgroundColor": PLOT_BG_COLOR,
        "minHeight": "100vh"
    })


def auto_arima_params(ts_data, max_p=3, max_d=2, max_q=3):
    """Otomatik ARIMA parametre seçimi"""
    best_aic = float('inf')
    best_params = None

    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(ts_data, order=(p, d, q))
                    fitted_model = model.fit()
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_params = (p, d, q)
                except:
                    continue

    return best_params, best_aic


def check_stationarity(ts_data):
    """ADF testi ile durağanlık kontrolü"""
    result = adfuller(ts_data.dropna())

    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4],
        'is_stationary': result[1] < 0.05
    }


@dash.callback(
    Output('arima-full-graph', 'figure'),
    Output('arima-decomposition-graph', 'figure'),
    Output('arima-7day-graph', 'figure'),
    Output('arima-30day-graph', 'figure'),
    Output('arima-metrics-output', 'children'),
    Output('stationarity-results', 'children'),
    Input('arima-product-select', 'value')
)
def update_arima_forecast(product_id):
    if not product_id:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Lütfen bir ürün seçiniz",
            plot_bgcolor=PLOT_BG_COLOR,
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12)
        )
        return empty_fig, empty_fig, empty_fig, empty_fig, "", ""

    # Veri hazırlama
    dff = df[df['Product_ID'] == product_id].copy()
    daily_demand = dff.groupby('Date')['Demand'].sum().reset_index()
    daily_demand.set_index('Date', inplace=True)
    daily_demand = daily_demand.asfreq('D', fill_value=0)

    ts_data = daily_demand['Demand']

    # Durağanlık testi
    stationarity_result = check_stationarity(ts_data)

    # Veri dönüşümü (gerekirse)
    if not stationarity_result['is_stationary']:
        ts_diff = ts_data.diff().dropna()
        stationarity_after_diff = check_stationarity(ts_diff)
    else:
        ts_diff = ts_data
        stationarity_after_diff = stationarity_result

    # Otomatik ARIMA parametre seçimi
    best_params, best_aic = auto_arima_params(ts_diff)

    # ARIMA model
    try:
        model = ARIMA(ts_data, order=best_params)
        fitted_model = model.fit()

        # Tahminler
        forecast_30 = fitted_model.forecast(steps=30)
        forecast_7 = forecast_30[:7]

        # In-sample tahminler
        fitted_values = fitted_model.fittedvalues

        # Model performansı
        y_true = ts_data[fitted_values.index]
        y_pred = fitted_values

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        r2 = r2_score(y_true, y_pred)
        avg_demand = y_true.mean()
        rmse_pct = (rmse / avg_demand) * 100

    except Exception as e:
        # Hata durumunda basit model kullan
        model = ARIMA(ts_data, order=(1, 1, 1))
        fitted_model = model.fit()
        forecast_30 = fitted_model.forecast(steps=30)
        forecast_7 = forecast_30[:7]
        fitted_values = fitted_model.fittedvalues

        y_true = ts_data[fitted_values.index]
        y_pred = fitted_values

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        r2 = r2_score(y_true, y_pred)
        avg_demand = y_true.mean()
        rmse_pct = (rmse / avg_demand) * 100
        best_params = (1, 1, 1)
        best_aic = fitted_model.aic

    # Durağanlık sonuçları HTML
    stationarity_html = html.Div([
        html.Div([
            html.H5("🔍 Durağanlık Test Sonuçları", style={'color': PRIMARY_COLOR, 'marginBottom': '15px'}),
            html.Div([
                html.Div([
                    html.P(f"📊 ADF Test İstatistiği: {stationarity_result['adf_statistic']:.4f}",
                           style={'fontSize': '1em', 'marginBottom': '5px'}),
                    html.P(f"📈 P-Value: {stationarity_result['p_value']:.4f}",
                           style={'fontSize': '1em', 'marginBottom': '5px'}),
                    html.P(
                        f"🎯 Durağanlık Durumu: {'✅ Durağan' if stationarity_result['is_stationary'] else '❌ Durağan Değil'}",
                        style={'fontSize': '1.1em', 'fontWeight': 'bold', 'marginBottom': '10px',
                               'color': SUCCESS_COLOR if stationarity_result['is_stationary'] else DANGER_COLOR}),
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                html.Div([
                    html.P(f"🔧 Seçilen ARIMA Parametreleri: {best_params}",
                           style={'fontSize': '1.1em', 'fontWeight': 'bold', 'marginBottom': '5px'}),
                    html.P(f"📊 AIC Değeri: {best_aic:.2f}",
                           style={'fontSize': '1em', 'marginBottom': '5px'}),
                    html.P("📈 p: Otoregresif bileşen", style={'fontSize': '0.9em', 'marginBottom': '3px'}),
                    html.P("📉 d: Fark alma derecesi", style={'fontSize': '0.9em', 'marginBottom': '3px'}),
                    html.P("⚡ q: Hareketli ortalama bileşeni", style={'fontSize': '0.9em', 'marginBottom': '3px'}),
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'})
            ])
        ], style={
            'background': 'rgba(255,255,255,0.9)',
            'padding': '20px',
            'borderRadius': '10px',
            'border': f'2px solid {ACCENT_COLOR}'
        })
    ])

    # Enhanced metrics with styling
    def yorum_renk(m, good, mid):
        if m < good:
            return SUCCESS_COLOR
        elif m < mid:
            return WARNING_COLOR
        return DANGER_COLOR

    yorumlar = html.Div([
        html.Div([
            html.Span("📈 RMSE / Ortalama Talep: ", style={'fontWeight': 'bold'}),
            html.Span(f"%{rmse_pct:.2f}", style={'fontSize': '1.2em', 'fontWeight': 'bold'}),
            html.Div(
                "✅ ARIMA Mükemmel performans" if rmse_pct < 20 else
                "⚠️ ARIMA Kabul edilebilir performans" if rmse_pct < 40 else
                "❌ ARIMA Performans iyileştirmesi gerekli",
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
                "✅ ARIMA Çok iyi doğruluk" if mape < 10 else
                "⚠️ ARIMA Kabul edilebilir doğruluk" if mape < 20 else
                "❌ ARIMA Doğruluk iyileştirmesi gerekli",
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
            html.Span("📊 R² Determination Skoru: ", style={'fontWeight': 'bold'}),
            html.Span(f"{r2:.4f}", style={'fontSize': '1.2em', 'fontWeight': 'bold'}),
            html.Div(
                "✅ ARIMA Güçlü model uyumu" if r2 >= 0.7 else
                "⚠️ ARIMA Orta seviye model uyumu" if r2 >= 0.5 else
                "❌ ARIMA Zayıf model uyumu",
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
                html.H5("📈 ARIMA Temel İstatistikler", style={'color': PRIMARY_COLOR, 'marginBottom': '15px'}),
                html.P(f"🎯 Seçilen Model: ARIMA{best_params}",
                       style={'fontSize': '1.1em', 'marginBottom': '8px', 'fontWeight': 'bold', 'color': ACCENT_COLOR}),
                html.P(f"📊 AIC Değeri: {best_aic:.2f}", style={'fontSize': '1.1em', 'marginBottom': '8px'}),
                html.P(f"📈 Ortalama Talep: {avg_demand:,.2f} adet", style={'fontSize': '1.1em', 'marginBottom': '8px'}),
                html.P(f"📊 RMSE: {rmse:,.2f}", style={'fontSize': '1.1em', 'marginBottom': '8px'}),
                html.P(f"🎯 MAPE: %{mape:.2f}", style={'fontSize': '1.1em', 'marginBottom': '8px'}),
                html.P(f"📈 R² Skoru: {r2:.4f}", style={'fontSize': '1.1em', 'marginBottom': '15px'}),
            ], style={'marginBottom': '20px'}),
            html.Hr(style={'border': f'1px solid {ACCENT_COLOR}', 'margin': '20px 0'}),
            html.H5("🎯 ARIMA Performans Değerlendirmesi", style={'color': PRIMARY_COLOR, 'marginBottom': '15px'}),
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

    # 1. Ana ARIMA Grafiği
    fig_full = go.Figure()

    # Gerçek veriler
    fig_full.add_trace(go.Scatter(
        x=ts_data.index,
        y=ts_data.values,
        mode='lines+markers',
        name='📊 Gerçek Talep',
        line=dict(color=PRIMARY_COLOR, width=2),
        marker=dict(size=4)
    ))

    # ARIMA fitted values
    fig_full.add_trace(go.Scatter(
        x=fitted_values.index,
        y=fitted_values.values,
        mode='lines',
        name='📈 ARIMA Modeli (Geçmiş)',
        line=dict(color=ACCENT_COLOR, width=3)
    ))

    # Gelecek tahminleri
    future_dates = pd.date_range(start=ts_data.index[-1] + pd.Timedelta(days=1), periods=30)

    fig_full.add_trace(go.Scatter(
        x=future_dates,
        y=forecast_30.values,
        mode='lines+markers',
        name='🚀 ARIMA Tahmin (30 Gün)',
        line=dict(color=SECONDARY_COLOR, width=4, dash='dot'),
        marker=dict(size=8, symbol='diamond')
    ))

    fig_full = style_graph(fig_full, f"📈 {product_id} Ürünü - ARIMA ile Tahmin Analizi")
    fig_full.update_xaxes(title="📅 Tarih")
    fig_full.update_yaxes(title="📊 Talep Miktarı")

    # 2. Seasonal Decomposition
    try:
        decomposition = seasonal_decompose(ts_data, model='additive', period=7)

        fig_decomp = make_subplots(
            rows=4, cols=1,
            subplot_titles=('📊 Orijinal Seri', '📈 Trend', '🔄 Mevsimsel', '📉 Residual'),
            vertical_spacing=0.08
        )

        # Orijinal seri
        fig_decomp.add_trace(
            go.Scatter(x=ts_data.index, y=ts_data.values, name='Orijinal', line=dict(color=PRIMARY_COLOR)),
            row=1, col=1
        )

        # Trend
        fig_decomp.add_trace(
            go.Scatter(x=decomposition.trend.index, y=decomposition.trend.values, name='Trend',
                       line=dict(color=ACCENT_COLOR)),
            row=2, col=1
        )

        # Mevsimsel
        fig_decomp.add_trace(
            go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal.values, name='Mevsimsel',
                       line=dict(color=SUCCESS_COLOR)),
            row=3, col=1
        )

        # Residual
        fig_decomp.add_trace(
            go.Scatter(x=decomposition.resid.index, y=decomposition.resid.values, name='Residual',
                       line=dict(color=DANGER_COLOR)),
            row=4, col=1
        )

        fig_decomp.update_layout(
            title=dict(
                text="🔄 Zaman Serisi Ayrıştırma Analizi",
                font=dict(size=18, color=PRIMARY_COLOR),
                x=0.5
            ),
            plot_bgcolor=PLOT_BG_COLOR,
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12),
            height=600,
            showlegend=False
        )

        # Alt başlıkları büyüt
        fig_decomp.update_annotations(font_size=16)

    except:
        fig_decomp = go.Figure()
        fig_decomp.add_annotation(
            text="⚠️ Ayrıştırma analizi için yeterli veri yok",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color=DANGER_COLOR)
        )
        fig_decomp.update_layout(
            title="🔄 Zaman Serisi Ayrıştırma Analizi",
            height=400
        )

    # 3. 7 Günlük Tahmin
    fig_7d = go.Figure()
    future_7d_dates = future_dates[:7]

    fig_7d.add_trace(go.Scatter(
        x=future_7d_dates,
        y=forecast_7.values,
        mode='lines+markers',
        name='📅 7 Günlük ARIMA Tahmin',
        line=dict(color=SUCCESS_COLOR, width=3),
        marker=dict(size=10, symbol='diamond'),
        fill='tozeroy',
        fillcolor=f'rgba(39,174,96,0.2)'
    ))

    fig_7d = style_graph(fig_7d, "📅 7 Günlük ARIMA Detaylı Talep Tahmini")
    fig_7d.update_xaxes(title="📅 Tarih")
    fig_7d.update_yaxes(title="📊 Tahmin Edilen Talep")

    # 4. 30 Günlük Tahmin
    fig_30d = go.Figure()

    fig_30d.add_trace(go.Scatter(
        x=future_dates,
        y=forecast_30.values,
        mode='lines+markers',
        name='📊 30 Günlük ARIMA Tahmin',
        line=dict(color=WARNING_COLOR, width=3),
        marker=dict(size=6),
        fill='tozeroy',
        fillcolor=f'rgba(243,156,18,0.2)'
    ))

    fig_30d = style_graph(fig_30d, "📊 30 Günlük ARIMA Uzun Vadeli Talep Tahmini")
    fig_30d.update_xaxes(title="📅 Tarih")
    fig_30d.update_yaxes(title="📊 Tahmin Edilen Talep")

    return fig_full, fig_decomp, fig_7d, fig_30d, metrics, stationarity_html