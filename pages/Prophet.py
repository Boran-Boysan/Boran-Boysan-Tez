import dash
from dash import html, dcc, Input, Output
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

dash.register_page(__name__, path="/prophet_forecast", name="Prophet Zaman Serisi Tahmini")

df = pd.read_csv("demand_forecasting_data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['Product_ID'] = df['Product_ID'].astype('category')

# Design constants (aynı)
PLOT_BG_COLOR = '#ECF0F1'
PRIMARY_COLOR = '#2C3E50'
ACCENT_COLOR = '#E67E22'
SECONDARY_COLOR = '#3498DB'
SUCCESS_COLOR = '#27AE60'
WARNING_COLOR = '#F39C12'
DANGER_COLOR = '#E74C3C'

# Stil tanımlamaları (aynı)
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


def clean_data(data):
    """Veri temizleme fonksiyonu"""
    # Eksik değerleri temizle
    data = data.dropna()

    # Negatif değerleri temizle
    data = data[data['y'] > 0]

    # Outlier temizleme (IQR yöntemi)
    Q1 = data['y'].quantile(0.25)
    Q3 = data['y'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Outlier'ları temizle
    data_cleaned = data[(data['y'] >= lower_bound) & (data['y'] <= upper_bound)]

    return data_cleaned


def prepare_enhanced_data(df, product_id):
    """Gelişmiş veri hazırlama"""
    dff = df[df['Product_ID'] == product_id].copy()

    # Kolon adlarını temizle (boşlukları kaldır)
    dff.columns = dff.columns.str.strip()

    # Mevcut kolonları kontrol et
    available_cols = {}
    required_cols = ['Demand', 'Marketing_Effect', 'Price', 'Discount', 'Stock_Availability', 'Public_Holiday']

    for col in required_cols:
        if col in dff.columns:
            available_cols[col] = 'sum' if col == 'Demand' else ('max' if col == 'Public_Holiday' else 'mean')

    # Günlük aggregation - sadece mevcut kolonlarla
    daily_data = dff.groupby('Date').agg(available_cols).reset_index()

    # Prophet formatına çevir
    prophet_data = daily_data.rename(columns={'Date': 'ds', 'Demand': 'y'})

    return prophet_data, daily_data


def create_enhanced_holidays(df):
    """Gelişmiş tatil günleri"""
    holidays = df[df['Public_Holiday'] == 1]['Date'].unique()

    if len(holidays) == 0:
        return None

    holidays_df = pd.DataFrame({
        'holiday': 'public_holiday',
        'ds': pd.to_datetime(holidays),
        'lower_window': -1,  # Tatil öncesi etki
        'upper_window': 1,  # Tatil sonrası etki
    })

    return holidays_df


def layout():
    return html.Div([
        # Main Title
        html.Div([
            html.H2("📊 İyileştirilmiş Prophet ile Mevsimsel Zaman Serisi Tahmini", style={
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
                html.H4("🎛️ İyileştirilmiş Prophet Model Parametreleri", style={
                    'color': PRIMARY_COLOR,
                    'marginBottom': '20px',
                    'textAlign': 'center',
                    'borderBottom': f'2px solid {ACCENT_COLOR}',
                    'paddingBottom': '10px'
                }),
                html.Label("📊 Ürün Seçiniz:", style={
                    'fontWeight': 'bold',
                    'color': PRIMARY_COLOR,
                    'marginBottom': '10px',
                    'display': 'block',
                    'fontSize': '1.1em'
                }),
                dcc.Dropdown(
                    id='prophet-product-select',
                    options=[{'label': f'Ürün {i}', 'value': i} for i in sorted(df['Product_ID'].unique())],
                    placeholder="Bir ürün seçiniz...",
                    style={
                        'marginBottom': '15px',
                        'fontSize': '1rem'
                    }
                ),
                html.Div([
                    html.P("🔧 Prophet İyileştirmeleri:",
                           style={'fontWeight': 'bold', 'color': PRIMARY_COLOR, 'marginBottom': '5px'}),
                    html.Ul([
                        html.Li("🧹 Otomatik veri temizleme ve outlier kontrolü",
                                style={'fontSize': '0.9em', 'marginBottom': '3px'}),
                        html.Li("📊 Dış değişkenler (fiyat, pazarlama, stok, indirim)",
                                style={'fontSize': '0.9em', 'marginBottom': '3px'}),
                        html.Li("📅 Aylık ve üç aylık mevsimsellik eklendi",
                                style={'fontSize': '0.9em', 'marginBottom': '3px'}),
                        html.Li("🎉 Genişletilmiş tatil etkisi modelleme",
                                style={'fontSize': '0.9em', 'marginBottom': '3px'}),
                        html.Li("⚙️ Optimize edilmiş model parametreleri",
                                style={'fontSize': '0.9em', 'marginBottom': '3px'})
                    ], style={'paddingLeft': '20px', 'color': '#5D6D7E'})
                ], style={
                    'background': f'rgba(39, 174, 96, 0.05)',
                    'padding': '15px',
                    'borderRadius': '8px',
                    'marginTop': '10px',
                    'border': f'1px solid rgba(39, 174, 96, 0.2)'
                })
            ], style=filter_container_style)
        ]),

        # Aynı grafik bölümleri
        html.Div([
            html.Div("📊 İyileştirilmiş Prophet Genel Tahmin Analizi", style=graph_header_style),
            html.Div([
                dcc.Graph(id='prophet-full-graph')
            ], style=graph_content_style)
        ], style=graph_style, className='graph-hover'),

        html.Div([
            html.Div("🔍 İyileştirilmiş Prophet Bileşen Analizi", style=graph_header_style),
            html.Div([
                dcc.Graph(id='prophet-components-graph')
            ], style=graph_content_style)
        ], style=graph_style, className='graph-hover'),

        html.Div([
            html.Div("📅 7 Günlük İyileştirilmiş Prophet Tahmini", style=graph_header_style),
            html.Div([
                dcc.Graph(id='prophet-7day-graph')
            ], style=graph_content_style)
        ], style=graph_style, className='graph-hover'),

        html.Div([
            html.Div("📊 30 Günlük İyileştirilmiş Prophet Tahmini", style=graph_header_style),
            html.Div([
                dcc.Graph(id='prophet-30day-graph')
            ], style=graph_content_style)
        ], style=graph_style, className='graph-hover'),

        # Metrics Section
        html.Div([
            html.Div([
                html.H4("📊 İyileştirilmiş Prophet Model Performans Metrikleri", style={
                    'color': PRIMARY_COLOR,
                    'marginBottom': '20px',
                    'textAlign': 'center',
                    'borderBottom': f'2px solid {ACCENT_COLOR}',
                    'paddingBottom': '10px'
                }),
                html.Div(id="prophet-metrics-output", style={"fontSize": "1.1rem"})
            ], style=metrics_container_style)
        ])

    ], style={
        "padding": "30px",
        "backgroundColor": PLOT_BG_COLOR,
        "minHeight": "100vh"
    })


@dash.callback(
    Output('prophet-full-graph', 'figure'),
    Output('prophet-components-graph', 'figure'),
    Output('prophet-7day-graph', 'figure'),
    Output('prophet-30day-graph', 'figure'),
    Output('prophet-metrics-output', 'children'),
    Input('prophet-product-select', 'value')
)
def update_improved_prophet_forecast(product_id):
    if not product_id:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Lütfen bir ürün seçiniz",
            plot_bgcolor=PLOT_BG_COLOR,
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12)
        )
        return empty_fig, empty_fig, empty_fig, empty_fig, ""

    try:
        # Gelişmiş veri hazırlama
        prophet_data, daily_data = prepare_enhanced_data(df, product_id)

        # Veri temizleme
        prophet_data_cleaned = clean_data(prophet_data)

        # Minimum veri kontrolü
        if len(prophet_data_cleaned) < 30:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="Yetersiz veri - En az 30 günlük temiz veri gerekli")
            return empty_fig, empty_fig, empty_fig, empty_fig, "Yetersiz veri"

        # Gelişmiş tatil günleri
        holidays_df = create_enhanced_holidays(df)

        # İyileştirilmiş Prophet modeli
        model = Prophet(
            # Optimize edilmiş parametreler
            changepoint_prior_scale=0.01,  # Daha konservatif (0.05 -> 0.01)
            n_changepoints=15,  # Değişim noktası sayısı
            changepoint_range=0.8,  # Değişim noktası aralığı

            # Mevsimsellik
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            seasonality_prior_scale=0.1,  # Daha az esnek

            # Tatil ayarları
            holidays=holidays_df,
            holidays_prior_scale=0.1,

            # Diğer parametreler
            interval_width=0.80,  # Daha dar güven aralığı
            growth='linear'
        )

        # Ek mevsimsellik ekleme
        model.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=3,  # Daha az karmaşık
            mode='multiplicative'
        )

        # Quarterly seasonality
        model.add_seasonality(
            name='quarterly',
            period=365.25 / 4,
            fourier_order=2,  # Daha az karmaşık
            mode='multiplicative'
        )

        # Dış değişkenler ekleme - güvenli şekilde
        available_columns = prophet_data_cleaned.columns.tolist()
        print(f"Mevcut kolonlar: {available_columns}")  # Debug için

        added_regressors = []
        regressor_cols = ['Marketing_Effect', 'Price', 'Discount', 'Stock_Availability']

        for col in regressor_cols:
            if col in available_columns:
                try:
                    model.add_regressor(col, standardize=True)
                    added_regressors.append(col)
                    print(f"Eklenen regressor: {col}")
                except Exception as e:
                    print(f"Regressor {col} eklenirken hata: {e}")

        print(f"Toplam eklenen regressor sayısı: {len(added_regressors)}")

        # Model eğitimi
        model.fit(prophet_data_cleaned)

        # Gelecek tahminleri
        future = model.make_future_dataframe(periods=30)

        # Dış değişkenler için gelecek değerleri - güvenli şekilde
        for col in added_regressors:
            if col in prophet_data_cleaned.columns:
                # Son 30 günün ortalamasını kullan
                last_30_mean = prophet_data_cleaned[col].tail(30).mean()
                future[col] = last_30_mean  # Tüm future için aynı değer

        forecast = model.predict(future)

        # Model değerlendirme
        y_true = prophet_data_cleaned['y'].values
        y_pred = forecast['yhat'][:len(y_true)].values

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        r2 = r2_score(y_true, y_pred)
        avg_demand = y_true.mean()
        rmse_pct = (rmse / avg_demand) * 100

        # Residual analizi
        residuals = y_true - y_pred
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)

        # Outlier istatistikleri
        original_count = len(prophet_data)
        cleaned_count = len(prophet_data_cleaned)
        outliers_removed = original_count - cleaned_count

        # İyileştirilmiş metrikler
        def get_performance_color(metric, good_threshold, medium_threshold, is_higher_better=False):
            if is_higher_better:
                if metric >= good_threshold:
                    return SUCCESS_COLOR
                elif metric >= medium_threshold:
                    return WARNING_COLOR
                else:
                    return DANGER_COLOR
            else:
                if metric <= good_threshold:
                    return SUCCESS_COLOR
                elif metric <= medium_threshold:
                    return WARNING_COLOR
                else:
                    return DANGER_COLOR

        metrics = html.Div([
            html.Div([
                html.H5("📊 İyileştirilmiş Prophet Model Performansı",
                        style={'color': PRIMARY_COLOR, 'marginBottom': '20px'}),

                # Temel metrikler
                html.Div([
                    html.Div([
                        html.Span("🎯 MAPE: ", style={'fontWeight': 'bold'}),
                        html.Span(f"{mape:.2f}%", style={'fontSize': '1.3em', 'fontWeight': 'bold'}),
                        html.Div(
                            "✅ Mükemmel doğruluk" if mape < 10 else
                            "⚠️ İyi doğruluk" if mape < 20 else
                            "❌ Doğruluk iyileştirmesi gerekli",
                            style={'fontSize': '0.9em', 'marginTop': '5px', 'fontStyle': 'italic'}
                        )
                    ], style={
                        'backgroundColor': get_performance_color(mape, 10, 20),
                        'color': 'white',
                        'padding': '15px',
                        'borderRadius': '10px',
                        'marginBottom': '10px',
                        'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'
                    }),

                    html.Div([
                        html.Span("📊 RMSE Yüzdesi: ", style={'fontWeight': 'bold'}),
                        html.Span(f"{rmse_pct:.2f}%", style={'fontSize': '1.3em', 'fontWeight': 'bold'}),
                        html.Div(
                            "✅ Düşük hata oranı" if rmse_pct < 15 else
                            "⚠️ Orta hata oranı" if rmse_pct < 30 else
                            "❌ Yüksek hata oranı",
                            style={'fontSize': '0.9em', 'marginTop': '5px', 'fontStyle': 'italic'}
                        )
                    ], style={
                        'backgroundColor': get_performance_color(rmse_pct, 15, 30),
                        'color': 'white',
                        'padding': '15px',
                        'borderRadius': '10px',
                        'marginBottom': '10px',
                        'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'
                    }),

                    html.Div([
                        html.Span("📈 R² Skoru: ", style={'fontWeight': 'bold'}),
                        html.Span(f"{r2:.4f}", style={'fontSize': '1.3em', 'fontWeight': 'bold'}),
                        html.Div(
                            "✅ Güçlü model" if r2 >= 0.8 else
                            "⚠️ İyi model" if r2 >= 0.6 else
                            "❌ Zayıf model",
                            style={'fontSize': '0.9em', 'marginTop': '5px', 'fontStyle': 'italic'}
                        )
                    ], style={
                        'backgroundColor': get_performance_color(r2, 0.8, 0.6, is_higher_better=True),
                        'color': 'white',
                        'padding': '15px',
                        'borderRadius': '10px',
                        'marginBottom': '20px',
                        'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'
                    }),
                ]),

                # İyileştirme detayları
                html.Div([
                    html.H6("🔧 Uygulanan İyileştirmeler:", style={'color': PRIMARY_COLOR, 'marginBottom': '15px'}),
                    html.Div([
                        html.P(f"🧹 {outliers_removed} adet outlier temizlendi", style={'margin': '5px 0'}),
                        html.P(f"📊 {len(added_regressors)} adet dış değişken eklendi: {', '.join(added_regressors)}",
                               style={'margin': '5px 0'}),
                        html.P(f"📅 2 adet ek mevsimsellik (aylık + üç aylık)", style={'margin': '5px 0'}),
                        html.P(f"🎉 Genişletilmiş tatil etkisi (-1, +1 gün)", style={'margin': '5px 0'}),
                        html.P(f"⚙️ Optimize parametreler (changepoint_prior_scale: 0.01)", style={'margin': '5px 0'}),
                    ], style={'fontSize': '0.95em', 'color': '#5D6D7E'})
                ], style={
                    'backgroundColor': '#F8F9FA',
                    'padding': '15px',
                    'borderRadius': '8px',
                    'border': '1px solid #E9ECEF'
                }),

                # Residual analizi
                html.Div([
                    html.H6("📊 Residual Analizi:", style={'color': PRIMARY_COLOR, 'marginBottom': '15px'}),
                    html.P(f"📏 Residual Ortalama: {residual_mean:.3f}", style={'margin': '5px 0'}),
                    html.P(f"📊 Residual Std: {residual_std:.3f}", style={'margin': '5px 0'}),
                    html.P(
                        "✅ Bias yok" if abs(residual_mean) < 0.1 else "⚠️ Hafif bias var",
                        style={'margin': '5px 0', 'fontWeight': 'bold'}
                    ),
                ], style={
                    'backgroundColor': '#E8F5E8',
                    'padding': '15px',
                    'borderRadius': '8px',
                    'marginTop': '15px',
                    'border': '1px solid #C3E6C3'
                })
            ])
        ])

        # Grafik oluşturma fonksiyonu
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

        # Ana grafik
        fig_full = go.Figure()

        # Gerçek veriler
        fig_full.add_trace(go.Scatter(
            x=prophet_data_cleaned['ds'],
            y=prophet_data_cleaned['y'],
            mode='lines+markers',
            name='📊 Gerçek Talep',
            line=dict(color=PRIMARY_COLOR, width=2),
            marker=dict(size=4)
        ))

        # Prophet tahmini
        fig_full.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines',
            name='📊 İyileştirilmiş Prophet',
            line=dict(color=ACCENT_COLOR, width=3)
        ))

        # Belirsizlik aralıkları
        fig_full.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_upper'],
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False
        ))

        fig_full.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_lower'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            name='🔮 Güven Aralığı',
            fillcolor='rgba(230,126,34,0.2)'
        ))

        # Gelecek tahminleri
        future_start = prophet_data_cleaned['ds'].max()
        future_forecast = forecast[forecast['ds'] > future_start]

        fig_full.add_trace(go.Scatter(
            x=future_forecast['ds'],
            y=future_forecast['yhat'],
            mode='lines+markers',
            name='🚀 30 Günlük Tahmin',
            line=dict(color=SECONDARY_COLOR, width=4, dash='dot'),
            marker=dict(size=8, symbol='diamond')
        ))

        fig_full = style_graph(fig_full, f"📊 {product_id} - İyileştirilmiş Prophet Analizi")
        fig_full.update_xaxes(title="📅 Tarih")
        fig_full.update_yaxes(title="📊 Talep Miktarı")

        # Bileşenler grafiği
        fig_components = go.Figure()

        fig_components.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['trend'],
            mode='lines',
            name='📈 Trend',
            line=dict(color=PRIMARY_COLOR, width=3)
        ))

        fig_components.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['weekly'],
            mode='lines',
            name='📅 Haftalık',
            line=dict(color=SUCCESS_COLOR, width=2)
        ))

        fig_components.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yearly'],
            mode='lines',
            name='🌍 Yıllık',
            line=dict(color=WARNING_COLOR, width=2)
        ))

        if 'monthly' in forecast.columns:
            fig_components.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['monthly'],
                mode='lines',
                name='📊 Aylık',
                line=dict(color=SECONDARY_COLOR, width=2)
            ))

        fig_components = style_graph(fig_components, "🔍 İyileştirilmiş Prophet Bileşenleri")
        fig_components.update_xaxes(title="📅 Tarih")
        fig_components.update_yaxes(title="📊 Bileşen Değeri")

        # 7 günlük grafik
        fig_7d = go.Figure()
        future_7d = future_forecast.head(7)

        fig_7d.add_trace(go.Scatter(
            x=future_7d['ds'],
            y=future_7d['yhat'],
            mode='lines+markers',
            name='📅 7 Günlük Tahmin',
            line=dict(color=SUCCESS_COLOR, width=3),
            marker=dict(size=10, symbol='diamond'),
            fill='tozeroy',
            fillcolor=f'rgba(39,174,96,0.2)'
        ))

        fig_7d = style_graph(fig_7d, "📅 7 Günlük İyileştirilmiş Tahmin")
        fig_7d.update_xaxes(title="📅 Tarih")
        fig_7d.update_yaxes(title="📊 Tahmin Edilen Talep")

        # 30 günlük grafik
        fig_30d = go.Figure()

        fig_30d.add_trace(go.Scatter(
            x=future_forecast['ds'],
            y=future_forecast['yhat'],
            mode='lines+markers',
            name='📊 30 Günlük Tahmin',
            line=dict(color=WARNING_COLOR, width=3),
            marker=dict(size=6),
            fill='tozeroy',
            fillcolor=f'rgba(243,156,18,0.2)'
        ))

        fig_30d = style_graph(fig_30d, "📊 30 Günlük İyileştirilmiş Tahmin")
        fig_30d.update_xaxes(title="📅 Tarih")
        fig_30d.update_yaxes(title="📊 Tahmin Edilen Talep")

        return fig_full, fig_components, fig_7d, fig_30d, metrics

    except Exception as e:
        error_fig = go.Figure()
        error_fig.update_layout(
            title=f"Hata: {str(e)}",
            plot_bgcolor=PLOT_BG_COLOR,
            paper_bgcolor='white'
        )
        error_msg = html.Div([
            html.H5("❌ Hata Oluştu", style={'color': DANGER_COLOR}),
            html.P(f"Hata detayı: {str(e)}", style={'color': '#666'})
        ])
        return error_fig, error_fig, error_fig, error_fig, error_msg