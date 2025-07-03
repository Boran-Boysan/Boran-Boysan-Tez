import dash
from dash import Dash, html, dcc, Input, Output, State, page_container, dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import dash.exceptions

# External stylesheets (Bootstrap + Custom CSS)
external_stylesheets = [
    'https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
]

# Custom CSS for hover effects
custom_css = """
<style>
.menu-close-btn:hover {
    background-color: #E67E22 !important;
    color: white !important;
    transform: scale(1.1);
    box-shadow: 0 4px 15px rgba(230,126,34,0.5) !important;
}

.menu-toggle-btn:hover {
    background-color: #D35400 !important;
    transform: scale(1.05);
    box-shadow: 0 4px 20px rgba(230,126,34,0.4) !important;
}

.apply-btn:hover {
    background: linear-gradient(135deg, #D35400, #E67E22) !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(230,126,34,0.4) !important;
}

.menu-link:hover {
    background-color: rgba(230,126,34,0.2) !important;
    border: 1px solid #E67E22 !important;
    transform: translateX(5px);
}
</style>
"""

# Initialize the Dash app with pages support
app = Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    use_pages=True,
    pages_folder="pages",
    suppress_callback_exceptions=True
)
server = app.server

# Enhanced theme colors
PRIMARY_COLOR = '#2C3E50'
ACCENT_COLOR = '#E67E22'
SECONDARY_COLOR = '#3498DB'
SUCCESS_COLOR = '#27AE60'
WARNING_COLOR = '#F39C12'
DANGER_COLOR = '#E74C3C'
BACKGROUND_COLOR = '#ECF0F1'
TEXT_COLOR = '#FFFFFF'
PLOT_BG_COLOR = '#FAFAFA'

# Color palette for graphs
GRAPH_COLORS = ['#E67E22', '#3498DB', '#27AE60', '#9B59B6', '#F39C12', '#E74C3C', '#1ABC9C', '#34495E']

# Load and preprocess data
df = pd.read_csv("demand_forecasting_data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df[df['Demand'].notna() & (df['Demand'] > 0)].copy()
df['Year'] = df['Date'].dt.year
df['Week'] = df['Date'].dt.isocalendar().week
df['Month'] = df['Date'].dt.month
df['Month_Name'] = df['Date'].dt.strftime('%B')
df['Weekday'] = df['Date'].dt.day_name()
df['Quarter'] = df['Date'].dt.quarter
df['Public_Holiday'] = df['Public_Holiday'].astype(int)
df['Seasonal_Trend'] = df['Seasonal_Trend'].astype('category')
df['Marketing_Campaign'] = df['Marketing_Campaign'].astype('category')
df['Product_ID'] = df['Product_ID'].astype('category')

# Add derived metrics for better analysis
df['Revenue'] = df['Price'] * df['Demand']
df['Discount_Amount'] = df['Price'] * df['Discount']
df['Price_Category'] = pd.cut(df['Price'], bins=[0, 50, 100, 200, float('inf')],
                              labels=['Düşük', 'Orta', 'Yüksek', 'Premium'])


# Style definitions
def get_styles():
    return {
        'header': {
            'background': f'linear-gradient(135deg, {PRIMARY_COLOR}, #34495E)',
            'padding': '15px 25px',
            'color': TEXT_COLOR,
            'display': 'flex',
            'alignItems': 'center',
            'justifyContent': 'space-between',
            'borderBottom': f'4px solid {ACCENT_COLOR}',
            'boxShadow': '0 2px 10px rgba(0,0,0,0.1)',
            'position': 'sticky',
            'top': '0',
            'zIndex': '998'  # Menüden daha düşük z-index
        },
        'menu': {
            'background': f'linear-gradient(180deg, {PRIMARY_COLOR}, #34495E)',
            'padding': '25px',
            'width': '320px',
            'color': TEXT_COLOR,
            'boxShadow': '4px 0 15px rgba(0,0,0,0.2)',
            'borderRight': f'4px solid {ACCENT_COLOR}',
            'position': 'fixed',
            'top': '0',  # Header'dan başlayacak şekilde
            'left': '-320px',
            'height': '100vh',  # Tam ekran yüksekliği
            'transition': 'left 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
            'zIndex': '1000',
            'overflowY': 'auto',
            'overflowX': 'hidden',
            'paddingTop': '100px'  # Header için boşluk
        },
        'sidebar': {
            'background': f'linear-gradient(180deg, {PRIMARY_COLOR}, #34495E)',
            'padding': '15px',  # Padding'i azalt
            'width': '320px',
            'color': TEXT_COLOR,
            'overflowY': 'auto',
            'borderRight': f'4px solid {ACCENT_COLOR}',
            'boxShadow': '2px 0 10px rgba(0,0,0,0.1)',
            'minHeight': '100vh'
        },
        'content': {
            'backgroundColor': BACKGROUND_COLOR,
            'padding': '30px',
            'flex': '1',
            'minHeight': '100vh',
            'position': 'relative'
        },
        'content_full': {
            'backgroundColor': BACKGROUND_COLOR,
            'padding': '30px',
            'width': '100%',
            'minHeight': '100vh',
            'position': 'relative'
        },
        'graph_box': {
            'boxShadow': '0 8px 32px rgba(0,0,0,0.1)',
            'borderRadius': '15px',
            'marginBottom': '30px',
            'backgroundColor': '#fff',
            'overflow': 'hidden',
            'transition': 'all 0.3s ease',
            'border': '1px solid #e0e0e0'
        }
    }


styles = get_styles()


# Enhanced menu builder
def build_menu():
    def link_item(page):
        return dcc.Link(
            page.get('name', page['path']),
            href=page['path'],
            style={
                'display': 'block',
                'color': TEXT_COLOR,
                'padding': '12px 15px',
                'textDecoration': 'none',
                'borderRadius': '8px',
                'margin': '5px 0',
                'transition': 'all 0.3s ease',
                'border': '1px solid transparent'
            },
            className='menu-link'
        )

    prediction_pages = []
    other_pages = []
    for page in dash.page_registry.values():
        if "tahmin" in page["name"].lower() or "regresyon" in page["name"].lower() or "lstm" in page[
            "name"].lower() or "xgboost" in page["name"].lower() or "çoklu" in page["name"].lower():
            prediction_pages.append(page)
        else:
            other_pages.append(page)

    return html.Div([
        # Menü kapatma butonu
        html.Div([
            html.Button(
                "✕",
                id="menu-close",
                n_clicks=0,
                style={
                    'background': 'transparent',
                    'color': TEXT_COLOR,
                    'border': f'2px solid {ACCENT_COLOR}',
                    'fontSize': '1.5rem',
                    'padding': '8px 12px',
                    'cursor': 'pointer',
                    'borderRadius': '8px',
                    'transition': 'all 0.3s ease',
                    'float': 'right',
                    'marginBottom': '15px',
                    'width': '45px',
                    'height': '45px',
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'center',
                    'boxShadow': '0 2px 8px rgba(230,126,34,0.3)'
                },
                title="Menüyü Kapat",
                className='menu-close-btn'
            )
        ], style={'textAlign': 'right', 'marginBottom': '10px'}),

        html.H4('📊 Dashboard Bölümleri', style={
            'marginBottom': '20px',
            'color': TEXT_COLOR,
            'textAlign': 'center',
            'borderBottom': f'2px solid {ACCENT_COLOR}',
            'paddingBottom': '10px'
        }),
        html.H6('📈 Veri Görselleştirme', style={
            'color': ACCENT_COLOR,
            'marginTop': '20px',
            'fontWeight': 'bold',
            'fontSize': '1.1em'
        }),
        html.Ul([html.Li(link_item(p)) for p in other_pages],
                style={'listStyleType': 'none', 'paddingLeft': '0px'}),
        html.H6('🔮 Tahmin Modelleri', style={
            'color': ACCENT_COLOR,
            'marginTop': '25px',
            'fontWeight': 'bold',
            'fontSize': '1.1em'
        }),
        html.Ul([html.Li(link_item(p)) for p in prediction_pages],
                style={'listStyleType': 'none', 'paddingLeft': '0px'})
    ])


# Enhanced header
header = html.Div([
    html.Div([
        html.Button(
            "☰",
            id="menu-toggle",
            n_clicks=0,
            style={
                'background': ACCENT_COLOR,
                'color': 'white',
                'border': 'none',
                'fontSize': '1.8rem',
                'padding': '8px 12px',
                'cursor': 'pointer',
                'marginRight': '15px',
                'borderRadius': '8px',
                'transition': 'all 0.3s ease',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.2)'
            },
            className='menu-toggle-btn'
        ),
        html.Div([
            html.Img(src='/assets/icon.png', style={'height': '45px'}),
            html.Span("🚀 Tahmin Eden Tedarik Zinciri", style={
                'fontSize': '1.6rem',
                'marginLeft': '15px',
                'fontWeight': '600'
            })
        ], style={'display': 'flex', 'alignItems': 'center'})
    ], style={'display': 'flex', 'alignItems': 'center'})
], style=styles['header'])


sidebar = html.Div([
    html.H4("🎛️ Akıllı Filtreler", style={
        'marginBottom': '15px',
        'color': TEXT_COLOR,
        'textAlign': 'center',  # Başlık ortala
        'borderBottom': f'2px solid {ACCENT_COLOR}',
        'paddingBottom': '8px'
    }),

    # Date Range Filter
    html.Div([
        html.Label(
            "📅 Tarih Aralığı",
            style={
                'color': TEXT_COLOR,
                'marginBottom': '8px',
                'display': 'block',
                'fontWeight': 'bold',
                'textAlign': 'center'  # Label ortala
            }
        ),
        html.Div([
            dcc.DatePickerRange(
                id='date-range',
                start_date=df['Date'].min(),
                end_date=df['Date'].max(),
                display_format='DD.MM.YYYY',
                calendar_orientation='vertical',
                number_of_months_shown=1,
                day_size=30,
                with_portal=False,
                style={'width': '100%'}
            )
        ], className='date-picker-container', style={'textAlign': 'center'})
    ], className='filter-container'),

    # Product ID Filter
    html.Div([
        html.Label("🏷️ Ürün ID", style={
            'color': TEXT_COLOR,
            'marginBottom': '8px',
            'display': 'block',
            'fontWeight': 'bold',
            'textAlign': 'center'  # Label ortala
        }),
        html.Div([
            dcc.Dropdown(
                id='product-dropdown',
                options=[{'label': f'Ürün {i}', 'value': i} for i in sorted(df['Product_ID'].unique())],
                multi=True,
                placeholder='Ürün seçiniz...',
                style={'color': 'black', 'width': '100%'}
            )
        ], style={'textAlign': 'center'})
    ], className='filter-container'),

    # Marketing Campaign Filter
    html.Div([
        html.Label("📢 Pazarlama Kampanyası", style={
            'color': TEXT_COLOR,
            'marginBottom': '8px',
            'display': 'block',
            'fontWeight': 'bold',
            'textAlign': 'center'
        }),
        html.Div([
            dcc.Dropdown(
                id='marketing-dropdown',
                options=[{'label': str(i).title(), 'value': i} for i in
                         sorted(df['Marketing_Campaign'].dropna().unique())],
                multi=True,
                placeholder='Kampanya seçiniz...',
                style={'color': 'black', 'width': '100%'}
            )
        ], style={'textAlign': 'center'})
    ], className='filter-container'),

    # Price Range Filter
    html.Div([
        html.Label("💰 Fiyat Aralığı", style={
            'color': TEXT_COLOR,
            'marginBottom': '8px',
            'display': 'block',
            'fontWeight': 'bold',
            'textAlign': 'center'
        }),
        html.Div([
            dcc.RangeSlider(
                id='price-slider',
                min=0,
                max=int(df['Price'].max()),
                step=10,
                value=[0, int(df['Price'].max())],
                marks={i: f'{i}₺' for i in range(0, int(df['Price'].max()) + 1, int(df['Price'].max()) // 5)})
        ], style={'margin': '10px 0'})
    ], className='filter-container'),

    # Discount Filter
    html.Div([
        html.Label("🎯 İndirim Oranı", style={
            'color': TEXT_COLOR,
            'marginBottom': '8px',
            'display': 'block',
            'fontWeight': 'bold',
            'textAlign': 'center'
        }),
        html.Div([
            dcc.RangeSlider(
                id='discount-slider',
                min=0,
                max=100,
                step=1,
                value=[0, 100],
                tooltip={"placement": "bottom", "always_visible": True},
                marks={i: f'{i}%' for i in range(0, 101, 25)}
            )
        ], style={'margin': '10px 0'})
    ], className='filter-container'),

    # Feature Selection
    html.Div([
        html.Label("📊 Analiz Özelliği", style={
            'color': TEXT_COLOR,
            'marginBottom': '8px',
            'display': 'block',
            'fontWeight': 'bold',
            'textAlign': 'center'
        }),
        html.Div([
            dcc.Dropdown(
                id='feature-select',
                options=[
                    {"label": "💰 Fiyat", "value": "Price"},
                    {"label": "🎯 İndirim", "value": "Discount"},
                    {"label": "📢 Pazarlama Etkisi", "value": "Marketing_Effect"},
                    {"label": "🏪 Rakip Fiyatı", "value": "Competitor_Price"}
                ],
                value='Price',
                style={'color': 'black', 'width': '100%'}
            )
        ], style={'textAlign': 'center'})
    ], className='filter-container'),

    # Apply Button
    html.Div([
        html.Button("✨ Filtreleri Uygula", id='apply-button', style={
            'marginTop': '15px',
            'width': '100%',
            'background': f'linear-gradient(135deg, {ACCENT_COLOR}, #D35400)',
            'color': 'white',
            'border': 'none',
            'padding': '12px',
            'borderRadius': '8px',
            'cursor': 'pointer',
            'fontSize': '1.1em',
            'fontWeight': 'bold',
            'transition': 'all 0.3s ease',
            'boxShadow': '0 4px 15px rgba(230,126,34,0.3)'
        }, className='apply-btn')
    ], style={'textAlign': 'center'})
], style=styles['sidebar'], className='sidebar-content')

# Enhanced main layout with better KPI cards
main_layout = html.Div([
    html.Div([
        html.H2("📈 Akıllı Dashboard - Tedarik Zinciri Analizi", style={
            'textAlign': 'center',
            'color': PRIMARY_COLOR,
            'marginBottom': '30px',
            'fontWeight': 'bold',
            'textShadow': '2px 2px 4px rgba(0,0,0,0.1)'
        }),

        # Enhanced KPI Cards
        html.Div([
            html.Div([
                html.Div([
                    html.I(className="fas fa-chart-line", style={'fontSize': '2rem', 'marginBottom': '10px'}),
                    html.H3("Ortalama Talep", style={'margin': '0', 'fontSize': '1.2rem'}),
                    html.H1(id="avg-demand", style={'margin': '10px 0', 'fontSize': '2.5rem', 'fontWeight': 'bold'})
                ], style={'textAlign': 'center'})
            ], className='kpi-card', style={
                "background": f"linear-gradient(135deg, {ACCENT_COLOR}, #D35400)",
                "padding": "25px",
                "color": "white",
                "borderRadius": "15px",
                "width": "30%",
                "minHeight": "140px",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center"
            }),

            html.Div([
                html.Div([
                    html.I(className="fas fa-money-bill-wave", style={'fontSize': '2rem', 'marginBottom': '10px'}),
                    html.H3("Ortalama Fiyat", style={'margin': '0', 'fontSize': '1.2rem'}),
                    html.H1(id="avg-price", style={'margin': '10px 0', 'fontSize': '2.5rem', 'fontWeight': 'bold'})
                ], style={'textAlign': 'center'})
            ], className='kpi-card', style={
                "background": f"linear-gradient(135deg, {SECONDARY_COLOR}, #2980B9)",
                "padding": "25px",
                "color": "white",
                "borderRadius": "15px",
                "width": "30%",
                "minHeight": "140px",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center"
            }),

            html.Div([
                html.Div([
                    html.I(className="fas fa-percentage", style={'fontSize': '2rem', 'marginBottom': '10px'}),
                    html.H3("Ortalama İndirim", style={'margin': '0', 'fontSize': '1.2rem'}),
                    html.H1(id="avg-discount", style={'margin': '10px 0', 'fontSize': '2.5rem', 'fontWeight': 'bold'})
                ], style={'textAlign': 'center'})
            ], className='kpi-card', style={
                "background": f"linear-gradient(135deg, {SUCCESS_COLOR}, #229954)",
                "padding": "25px",
                "color": "white",
                "borderRadius": "15px",
                "width": "30%",
                "minHeight": "140px",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center"
            })
        ], style={
            "display": "flex",
            "gap": "25px",
            "flexWrap": "wrap",
            "justifyContent": "center",
            "marginBottom": "40px"
        })
    ]),

    # Enhanced Graph Containers
    html.Div([
        html.Div("📊 Özellik-Talep İlişki Analizi", className='graph-header'),
        html.Div([dcc.Graph(id="feature-scatter")], className='graph-content')
    ], className='graph-container'),

    html.Div([
        html.Div("📅 Aylık Talep Trend Analizi", className='graph-header'),
        html.Div([dcc.Graph(id="monthly-demand")], className='graph-content')
    ], className='graph-container'),

    html.Div([
        html.Div("🎉 Tatil Günleri Talep Analizi", className='graph-header'),
        html.Div([dcc.Graph(id="holiday-demand")], className='graph-content')
    ], className='graph-container'),

    html.Div([
        html.Div("🏆 En Çok Satan Ürünler Sıralaması", className='graph-header'),
        html.Div([
            dash_table.DataTable(
                id="top-products-table",
                columns=[
                    {"name": "🏷️ Ürün ID", "id": "Product_ID"},
                    {"name": "📈 Toplam Talep", "id": "Demand", "type": "numeric", "format": {"specifier": ",.0f"}},
                    {"name": "💰 Toplam Gelir", "id": "Revenue", "type": "numeric", "format": {"specifier": ",.2f"}}
                ],
                style_cell={
                    "textAlign": "center",
                    "padding": "12px",
                    "fontFamily": "Arial, sans-serif"
                },
                style_header={
                    "fontWeight": "bold",
                    "backgroundColor": PRIMARY_COLOR,
                    "color": "white",
                    "fontSize": "1.1em",
                    "padding": "15px"
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 0},
                        'backgroundColor': '#FFD700',
                        'color': 'black',
                        'fontWeight': 'bold'
                    },
                    {
                        'if': {'row_index': 1},
                        'backgroundColor': '#C0C0C0',
                        'color': 'black',
                        'fontWeight': 'bold'
                    },
                    {
                        'if': {'row_index': 2},
                        'backgroundColor': '#CD7F32',
                        'color': 'black',
                        'fontWeight': 'bold'
                    }
                ],
                style_table={"width": "100%"},
                page_size=10
            )
        ], className='graph-content')
    ], className='graph-container enhanced-table')
])

# Register the main page
dash.register_page("main", path="/", name="Veri Keşfi ve Görselleştirme", layout=main_layout)

# App wrapper layout
template = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='menu-open', data=False),
    header,
    html.Div([
        html.Div(id="menu-container"),
        html.Div(id="sidebar-container"),  # Sidebar için ayrı container
        html.Div(page_container, id="page-container", style=styles['content'])
    ], style={'display': 'flex', 'flex': 1, 'position': 'relative'})
], style={
    'display': 'flex',
    'flexDirection': 'column',
    'minHeight': '100vh'
})

app.layout = template


@app.callback(
    Output("sidebar-container", "children"),
    Output("sidebar-container", "style"),
    Input("url", "pathname")
)
def update_sidebar_visibility(pathname):
    prediction_paths = [
        "/lineer_regresyon",
        "/lstm_forecast",
        "/polinom-ridge",
        "/xgboost_recursive",
        "/coklu_lineer_recursive"
    ]

    if pathname in prediction_paths:
        return html.Div(), {'display': 'none'}
    else:
        return sidebar, styles['sidebar']


@app.callback(
    Output("page-container", "style"),
    Input("url", "pathname")
)
def update_content_style(pathname):
    prediction_paths = [
        "/lineer_regresyon",
        "/lstm_forecast",
        "/polinom-ridge",
        "/xgboost_recursive",
        "/coklu_lineer_recursive"
    ]

    if pathname in prediction_paths:
        return styles['content_full']
    else:
        return styles['content']


# Enhanced callbacks for menu toggle
@app.callback(
    Output("menu-open", "data"),
    Input("menu-toggle", "n_clicks"),
    Input("menu-close", "n_clicks"),
    State("menu-open", "data"),
    prevent_initial_call=True
)
def toggle_menu_store(toggle_clicks, close_clicks, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_open

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == "menu-toggle":
        return not is_open
    elif trigger_id == "menu-close":
        return False  # Menüyü kapat

    return is_open


@app.callback(
    Output("menu-container", "children"),
    Output("menu-container", "style"),
    Input("menu-open", "data"),
    Input("url", "pathname")
)
def update_menu(is_open, _):
    style = styles['menu'].copy()

    if is_open:
        style['left'] = '0px'
        style['boxShadow'] = '4px 0 50px rgba(0,0,0,0.5)'
        menu_content = html.Div([
            html.Div(
                id='menu-overlay',
                style={
                    'position': 'fixed',
                    'top': '0',
                    'left': '320px',
                    'width': 'calc(100vw - 320px)',
                    'height': '100vh',
                    'backgroundColor': 'rgba(0,0,0,0.3)',
                    'zIndex': '999',
                    'cursor': 'pointer'
                }
            ),
            html.Div(build_menu(), style={
                'position': 'relative',
                'zIndex': '1001',
                'height': '100%',
                'overflowY': 'auto'
            })
        ])
    else:
        style['left'] = '-320px'
        style['boxShadow'] = '4px 0 15px rgba(0,0,0,0.2)'
        menu_content = build_menu()

    return menu_content, style


@app.callback(
    Output("menu-open", "data", allow_duplicate=True),
    Input("menu-overlay", "n_clicks"),
    prevent_initial_call=True
)
def close_menu_on_overlay_click(n_clicks):
    if n_clicks:
        return False
    return dash.no_update

@app.callback(
    Output("feature-scatter", "figure"),
    Output("monthly-demand", "figure"),
    Output("holiday-demand", "figure"),
    Output("top-products-table", "data"),
    Output("avg-demand", "children"),
    Output("avg-price", "children"),
    Output("avg-discount", "children"),
    Input("apply-button", "n_clicks"),
    State("feature-select", "value"),
    State("product-dropdown", "value"),
    State("marketing-dropdown", "value"),
    State("price-slider", "value"),
    State("discount-slider", "value"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
    State("url", "pathname")  # Pathname eklendi
)
def update_all(n_clicks, feature, product_ids, marketing, price_range, discount_range, start_date, end_date, pathname):
    prediction_paths = [
        "/lineer_regresyon",
        "/lstm_forecast",
        "/polinom-ridge",
        "/xgboost_recursive",
        "/coklu_lineer_recursive"
    ]

    if pathname in prediction_paths:
        raise dash.exceptions.PreventUpdate

    # Filter data
    filtered = df.copy()
    if product_ids:
        filtered = filtered[filtered['Product_ID'].isin(product_ids)]
    if marketing:
        filtered = filtered[filtered['Marketing_Campaign'].isin(marketing)]
    if price_range:
        filtered = filtered[(filtered['Price'] >= price_range[0]) & (filtered['Price'] <= price_range[1])]
    if discount_range:
        filtered = filtered[
            (filtered['Discount'] >= discount_range[0] / 100) & (filtered['Discount'] <= discount_range[1] / 100)]
    if start_date and end_date:
        filtered = filtered[(filtered['Date'] >= start_date) & (filtered['Date'] <= end_date)]

    feature_labels = {
        'Price': 'Fiyat Analizi',
        'Discount': 'İndirim Stratejisi',
        'Marketing_Effect': 'Pazarlama Etkisi',
        'Competitor_Price': 'Rakip Fiyat Analizi'
    }

    # Modern Color Palette for better visual appeal
    DISCRETE_COLORS = ['#667EEA', '#F093FB', '#4FACFE', '#43E97B', '#FA709A', '#764BA2']

    if feature == 'Price':
        #Premium styling
        price_stats = filtered[feature].describe()
        q25, q50, q75 = price_stats['25%'], price_stats['50%'], price_stats['75%']

        bins = [0, q25, q50, q75, float('inf')]
        labels = [
            f'💸 Ekonomik ≤{q25:.0f}₺',
            f'🛍️ Popüler {q25:.0f}-{q50:.0f}₺',
            f'💎 Premium {q50:.0f}-{q75:.0f}₺',
            f'👑 Lüks ≥{q75:.0f}₺'
        ]
        filtered['Category'] = pd.cut(filtered[feature], bins=bins, labels=labels)

    elif feature == 'Discount':
        # Modern approach
        discount_pct = filtered[feature] * 100

        bins = [0, 5, 15, 30, 100]
        labels = [
            '🎯 Tam Fiyat 0-5%',
            '🛍️ Cazip İndirim 5-15%',
            '🔥 Büyük Fırsat 15-30%',
            '💥 Mega İndirim 30%+'
        ]
        filtered['Category'] = pd.cut(discount_pct, bins=bins, labels=labels)

    elif feature == 'Marketing_Effect':
        #Creative labels
        bins = [0, 0.3, 0.6, 0.8, 1.0]
        labels = [
            '😴 Organik Büyüme',
            '📢 Aktif Kampanya',
            '🎺 Güçlü Mesajlaşma',
            '🚀 Viral Momentum'
        ]
        filtered['Category'] = pd.cut(filtered[feature], bins=bins, labels=labels)

    elif feature == 'Competitor_Price':
        # Strategic view
        competitor_avg = filtered[feature].mean()

        bins = [0, competitor_avg * 0.8, competitor_avg * 0.95, competitor_avg * 1.05, float('inf')]
        labels = [
            '🏆 Fiyat Lideri',
            '⚔️ Agresif Rekabet',
            '⚖️ Dengeli Rekabet',
            '🎩 Premium Konum'
        ]
        filtered['Category'] = pd.cut(filtered[feature], bins=bins, labels=labels)

    else:
        feature_stats = filtered[feature].describe()

        def create_elegant_labels(low, med, high):
            return [
                f'📉 Alt Performans ≤{med:.1f}',
                f'📊 Standart Performans {med:.1f}-{high:.1f}',
                f'📈 Üst Performans ≥{high:.1f}'
            ]

        try:
            filtered['Category'] = pd.qcut(
                filtered[feature],
                q=3,
                labels=create_elegant_labels(
                    feature_stats['min'],
                    feature_stats['50%'],
                    feature_stats['max']
                ),
                duplicates='drop'
            )
        except:
            filtered['Category'] = pd.cut(
                filtered[feature],
                bins=3,
                labels=create_elegant_labels(
                    feature_stats['min'],
                    feature_stats['50%'],
                    feature_stats['max']
                )
            )

    filtered_clean = filtered.dropna(subset=['Category'])

    if len(filtered_clean) == 0:
        # Boş veri durumu için özel grafik
        scatter_fig = go.Figure()
        scatter_fig.add_annotation(
            text="❌ Seçilen filtreler için veri bulunamadı veya kategorileme başarısız!",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="red"),
            bgcolor="rgba(255,240,240,0.8)",
            bordercolor="red",
            borderwidth=2,
            borderpad=10
        )
        scatter_fig.update_layout(
            title=f"❌ {feature_labels.get(feature, feature)} Analizi - Veri Bulunamadı",
            plot_bgcolor=PLOT_BG_COLOR,
            paper_bgcolor='white',
            height=500,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )

    elif len(filtered_clean['Category'].unique()) < 2:
        single_cat = filtered_clean['Category'].iloc[0]
        avg_demand = filtered_clean['Demand'].mean()

        scatter_fig = go.Figure(data=[
            go.Bar(x=[single_cat], y=[avg_demand],
                   marker_color=DISCRETE_COLORS[0],
                   text=[f'{avg_demand:.0f}'],
                   textposition='outside')
        ])

        scatter_fig.update_layout(
            title=f"📊 {feature_labels.get(feature, feature)} • Tek Kategori Analizi",
            plot_bgcolor=PLOT_BG_COLOR,
            paper_bgcolor='white',
            height=500,
            xaxis=dict(title="Kategori"),
            yaxis=dict(title="Ortalama Talep")
        )

    else:
        scatter_fig = px.box(
            filtered_clean,
            x='Category',
            y='Demand',
            color='Category',
            title=f"📊 {feature_labels.get(feature, feature)} • Kategori Bazlı Talep Analizi",
            color_discrete_sequence=DISCRETE_COLORS,
            points="outliers",
            hover_data=['Product_ID', 'Date']
        )

        category_stats = []
        for cat in filtered_clean['Category'].unique():
            if pd.notna(cat):
                cat_data = filtered_clean[filtered_clean['Category'] == cat]['Demand']
                stats = {
                    'category': cat,
                    'median': cat_data.median(),
                    'mean': cat_data.mean(),
                    'count': len(cat_data),
                    'std': cat_data.std(),
                    'q25': cat_data.quantile(0.25),
                    'q75': cat_data.quantile(0.75)
                }
                category_stats.append(stats)

        # Enhanced hover template with more information
        scatter_fig.update_traces(
            hovertemplate='<b>%{x}</b><br>' +
                          '📊 Talep: %{y:,.0f}<br>' +
                          '🏷️ Ürün: %{customdata[0]}<br>' +
                          '📅 Tarih: %{customdata[1]}<br>' +
                          '<extra></extra>',
            marker=dict(
                size=5,
                opacity=0.7,
                line=dict(width=1, color='rgba(255,255,255,0.8)')
            ),
            line=dict(width=2.5)
        )

        # Add enhanced mean markers with animation effect
        for i, stats in enumerate(category_stats):
            cat = stats['category']
            mean_val = stats['mean']

            scatter_fig.add_trace(
                go.Scatter(
                    x=[cat],
                    y=[mean_val],
                    mode='markers',
                    marker=dict(
                        symbol='diamond',
                        size=15,
                        color=DISCRETE_COLORS[i % len(DISCRETE_COLORS)],
                        line=dict(width=3, color='white'),
                        opacity=0.9
                    ),
                    name=f'📊 Ortalama',
                    showlegend=True if i == 0 else False,
                    hovertemplate=f'<b>📊 İstatistiksel Özet</b><br>' +
                                  f'Kategori: {cat}<br>' +
                                  f'Ortalama: {mean_val:.0f}<br>' +
                                  f'Medyan: {stats["median"]:.0f}<br>' +
                                  f'Standart Sapma: {stats["std"]:.0f}<br>' +
                                  f'Veri Sayısı: {stats["count"]}<br>' +
                                  '<extra></extra>'
                )
            )

        # Ultra-modern layout with glassmorphism effect
        scatter_fig.update_layout(
            plot_bgcolor='rgba(248,250,252,0.8)',
            paper_bgcolor='white',
            font=dict(
                family="'Inter', 'Segoe UI', 'Arial', sans-serif",
                size=11,
                color='#2D3748'
            ),
            title=dict(
                font=dict(size=18, color='#1A202C', weight=700),
                x=0.5,
                y=0.95,
                pad=dict(b=20)
            ),
            xaxis=dict(
                title=dict(
                    text=f"🎯 {feature_labels.get(feature, feature)} Kategorileri",
                    font=dict(size=13, color='#4A5568', weight=600)
                ),
                tickangle=-45,
                tickfont=dict(size=10, color='#718096'),
                showgrid=False,
                showline=True,
                linewidth=2,
                linecolor='rgba(74,85,104,0.2)',
                automargin=True
            ),
            yaxis=dict(
                title=dict(
                    text="📈 Talep Miktarı Dağılımı",
                    font=dict(size=13, color='#4A5568', weight=600)
                ),
                tickfont=dict(size=10, color='#718096'),
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(226,232,240,0.8)',
                showline=True,
                linewidth=2,
                linecolor='rgba(74,85,104,0.2)',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='rgba(74,85,104,0.3)'
            ),
            height=550,
            margin=dict(l=70, r=40, t=90, b=120),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.20,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(226,232,240,0.8)",
                borderwidth=2,
                font=dict(size=11, color='#4A5568')
            ),
            annotations=[
                dict(
                    text=f"📊 <b>Toplam Veri:</b> {len(filtered_clean):,} • <b>Kategori:</b> {len(category_stats)} • <b>Ortalama Talep:</b> {filtered_clean['Demand'].mean():.0f}",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=1.05,
                    xanchor='center', yanchor='bottom',
                    font=dict(size=11, color='#4A5568'),
                    bgcolor="rgba(248,250,252,0.9)",
                    bordercolor="rgba(203,213,224,0.6)",
                    borderwidth=1,
                    borderpad=8
                )
            ],
            shapes=[
                dict(
                    type="rect",
                    xref="paper", yref="paper",
                    x0=0, y0=0, x1=1, y1=1,
                    fillcolor="rgba(255,255,255,0.1)",
                    line=dict(color="rgba(0,0,0,0.1)", width=1)
                )
            ]
        )

        # Add subtle animation on hover
        scatter_fig.update_traces(
            hoverlabel=dict(
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="rgba(102,126,234,0.8)",
                font=dict(size=12, color='#2D3748')
            )
        )

    # Enhanced Monthly Demand Chart
    monthly_agg = filtered.groupby("Month").agg({
        'Demand': 'sum',
        'Revenue': 'sum',
        'Price': 'mean'
    }).reset_index()

    monthly_agg['Month_Name'] = monthly_agg['Month'].map({
        1: 'Ocak', 2: 'Şubat', 3: 'Mart', 4: 'Nisan', 5: 'Mayıs', 6: 'Haziran',
        7: 'Temmuz', 8: 'Ağustos', 9: 'Eylül', 10: 'Ekim', 11: 'Kasım', 12: 'Aralık'
    })

    # Create subplot with secondary y-axis
    monthly_fig = make_subplots(
        specs=[[{"secondary_y": True}]],
        subplot_titles=["📅 Aylık Performans Analizi"]
    )

    # Add bar chart for demand
    monthly_fig.add_trace(
        go.Bar(
            x=monthly_agg['Month_Name'],
            y=monthly_agg['Demand'],
            name='Talep Miktarı',
            marker_color=ACCENT_COLOR,
            text=monthly_agg['Demand'],
            texttemplate='%{text:,.0f}',
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Talep: %{y:,.0f}<extra></extra>'
        ),
        secondary_y=False,
    )

    # Add line chart for revenue
    monthly_fig.add_trace(
        go.Scatter(
            x=monthly_agg['Month_Name'],
            y=monthly_agg['Revenue'],
            mode='lines+markers',
            name='Toplam Gelir',
            line=dict(color=SECONDARY_COLOR, width=3),
            marker=dict(size=8),
            hovertemplate='<b>%{x}</b><br>Gelir: %{y:,.2f}₺<extra></extra>'
        ),
        secondary_y=True,
    )

    monthly_fig.update_layout(
        plot_bgcolor=PLOT_BG_COLOR,
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=12),
        title=dict(font=dict(size=18, color=PRIMARY_COLOR), x=0.5),
        xaxis=dict(title="Aylar", title_font=dict(size=14, color=PRIMARY_COLOR)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        height=500,
        hovermode='x unified'
    )

    monthly_fig.update_yaxes(title_text="Talep Miktarı", secondary_y=False,
                             title_font=dict(size=14, color=PRIMARY_COLOR))
    monthly_fig.update_yaxes(title_text="Gelir (₺)", secondary_y=True,
                             title_font=dict(size=14, color=PRIMARY_COLOR))

    # Enhanced Holiday Analysis
    holiday_data = filtered.copy()
    holiday_data['Holiday_Status'] = holiday_data['Public_Holiday'].map({
        0: '📅 Normal Günler',
        1: '🎉 Tatil Günleri'
    })

    holiday_fig = go.Figure()

    # Add violin plot for better distribution visualization
    for i, status in enumerate(['📅 Normal Günler', '🎉 Tatil Günleri']):
        data_subset = holiday_data[holiday_data['Holiday_Status'] == status]['Demand']
        holiday_fig.add_trace(go.Violin(
            y=data_subset,
            name=status,
            box_visible=True,
            meanline_visible=True,
            fillcolor=GRAPH_COLORS[i],
            opacity=0.7,
            hovertemplate=f'<b>{status}</b><br>Talep: %{{y}}<extra></extra>'
        ))

    holiday_fig.update_layout(
        title=dict(
            text="🎉 Tatil Günleri vs Normal Günler Talep Dağılımı",
            font=dict(size=18, color=PRIMARY_COLOR),
            x=0.5
        ),
        plot_bgcolor=PLOT_BG_COLOR,
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=12),
        xaxis=dict(
            title=dict(
                text="Gün Türü",
                font=dict(size=14, color=PRIMARY_COLOR)
            )
        ),
        yaxis=dict(
            title=dict(
                text="Talep Miktarı",
                font=dict(size=14, color=PRIMARY_COLOR)
            )
        ),
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )

    # Enhanced Top Products Table
    top_products = (
        filtered.groupby("Product_ID")
        .agg({
            'Demand': 'sum',
            'Revenue': 'sum',
            'Price': 'mean'
        })
        .reset_index()
        .sort_values("Demand", ascending=False)
        .head(10)
    )

    top_products['Revenue'] = top_products['Revenue'].round(2)
    top_products_data = top_products.to_dict("records")

    # Enhanced KPI calculations
    avg_demand_val = f"{filtered['Demand'].mean():,.3f} adet"
    avg_price_val = f"{filtered['Price'].mean():.2f} ₺"
    avg_discount_val = f"{filtered['Discount'].mean() * 100:.1f}%"

    return (
        scatter_fig,
        monthly_fig,
        holiday_fig,
        top_products_data,
        avg_demand_val,
        avg_price_val,
        avg_discount_val
    )

if __name__ == '__main__':
    app.run(debug=True)

