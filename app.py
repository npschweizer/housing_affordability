import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

# McKinsey color scheme
MCKINSEY_COLORS = {
    'primary': '#0033A0',      # Deep blue
    'secondary': '#0066CC',    # Light blue
    'accent': '#FF6B35',       # Orange accent
    'success': '#00A652',      # Green
    'warning': '#FFB81C',      # Yellow
    'danger': '#DC3545',       # Red
    'dark': '#2C3E50',         # Dark gray
    'light': '#F8F9FA',        # Light gray
    'background': '#FFFFFF',   # White
    'text': '#2C3E50'          # Text color
}

# Custom CSS for McKinsey styling
mckinsey_css = f"""
<style>
    /* Main theme colors */
    .stApp {{
        background-color: {MCKINSEY_COLORS['background']};
        color: {MCKINSEY_COLORS['text']};
    }}
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {{
        color: {MCKINSEY_COLORS['primary']} !important;
        font-weight: 600;
    }}
    
    /* Primary buttons */
    .stButton > button {{
        background-color: {MCKINSEY_COLORS['primary']} !important;
        color: white !important;
        border: none;
        border-radius: 4px;
        font-weight: 500;
        transition: all 0.3s ease;
    }}
    
    .stButton > button:hover {{
        background-color: {MCKINSEY_COLORS['secondary']} !important;
    }}
    
    /* Secondary buttons */
    .stButton > button[kind="secondary"] {{
        background-color: {MCKINSEY_COLORS['light']} !important;
        color: {MCKINSEY_COLORS['primary']} !important;
        border: 1px solid {MCKINSEY_COLORS['primary']};
    }}
    
    /* Radio buttons */
    .stRadio > div > div > label {{
        background-color: {MCKINSEY_COLORS['light']};
        border: 1px solid {MCKINSEY_COLORS['primary']};
        border-radius: 4px;
        padding: 8px 12px;
        margin-right: 8px;
        color: {MCKINSEY_COLORS['text']};
    }}
    
    .stRadio > div > div > label[data-baseweb="radio-checked"] {{
        background-color: {MCKINSEY_COLORS['primary']};
        color: white;
    }}
    
    /* Select boxes */
    .stSelectbox > div > div > select {{
        background-color: {MCKINSEY_COLORS['light']};
        border: 1px solid {MCKINSEY_COLORS['primary']};
        border-radius: 4px;
        color: {MCKINSEY_COLORS['text']};
    }}
    
    /* Sliders - remove colored numbers */
    .stSlider > div > div > div > div {{
        background-color: {MCKINSEY_COLORS['primary']} !important;
    }}
    .stSlider > div > div > div > div > div {{
        background-color: {MCKINSEY_COLORS['secondary']} !important;
    }}
    .stSlider > div > div > div {{
        background-color: {MCKINSEY_COLORS['light']} !important;
    }}
    .stSlider [data-baseweb="slider"] {{
        color: #333333 !important;
    }}
    .stSlider [data-baseweb="slider-handle"] {{
        color: #333333 !important;
    }}
    
    /* Dataframes */
    .stDataFrame {{
        border: 1px solid {MCKINSEY_COLORS['light']};
        border-radius: 4px;
    }}
    
    /* Sidebar */
    .css-1d391kg {{
        background-color: {MCKINSEY_COLORS['light']};
    }}
    
    /* Tabs */
    .stTabs > div > div > div {{
        background-color: {MCKINSEY_COLORS['light']};
        border: 1px solid {MCKINSEY_COLORS['primary']};
    }}
    
    .stTabs > div > div > div[data-selected="true"] {{
        background-color: {MCKINSEY_COLORS['primary']};
        color: white;
    }}
    
    /* Expander */
    .stExpander > div > div > div > div {{
        background-color: {MCKINSEY_COLORS['light']};
        border: 1px solid {MCKINSEY_COLORS['primary']};
        border-radius: 4px;
    }}
    
    /* Info boxes */
    .stInfo {{
        background-color: {MCKINSEY_COLORS['light']};
        border-left: 4px solid {MCKINSEY_COLORS['primary']};
    }}
    
    .stSuccess {{
        background-color: #E8F5E8;
        border-left: 4px solid {MCKINSEY_COLORS['success']};
    }}
    
    .stWarning {{
        background-color: #FFF3CD;
        border-left: 4px solid {MCKINSEY_COLORS['warning']};
    }}
    
    .stError {{
        background-color: #F8D7DA;
        border-left: 4px solid {MCKINSEY_COLORS['danger']};
    }}
    
    /* Progress bar */
    .stProgress > div > div > div > div {{
        background-color: {MCKINSEY_COLORS['primary']};
    }}
    
    /* Charts */
    .stPlotlyChart {{
        border: 1px solid {MCKINSEY_COLORS['light']};
        border-radius: 4px;
    }}
</style>
"""

# Set page config
st.set_page_config(
    page_title="City Affordability Dashboard",
    page_icon="📊",
    layout="wide"
)

# Helper function to check if a column is a percentage column
def is_percentage_column(col_name):
    """Check if a column name represents a percentage value"""
    return 'percent' in col_name.lower() or 'pct' in col_name.lower()

# Helper function for variable description boxes
def create_description_box(variable_name, description):
    """Create a styled description box for a variable"""
    return f"""
    <div style="padding: 12px; background-color: {MCKINSEY_COLORS['light']}; border-radius: 6px; border-left: 4px solid {MCKINSEY_COLORS['secondary']}; margin-bottom: 10px;">
        <strong style="color: {MCKINSEY_COLORS['primary']}; font-size: 14px;">{variable_name}</strong><br>
        <span style="color: {MCKINSEY_COLORS['text']}; font-size: 12px;">{description}</span>
    </div>
    """

# Apply custom CSS
st.markdown(mckinsey_css, unsafe_allow_html=True)

# Zillow variable descriptions lookup
ZILLOW_VARIABLE_DESCRIPTIONS = {
    'NAME': 'Geographic area name (city, metro, or region identifier)',
    'TOT_POP': 'Total population of the geographic area',
    'TOT_MALE': 'Total male population in the geographic area',
    'TOT_FEMALE': 'Total female population in the geographic area',
    'PCT_MALE': 'Percentage of male population in the geographic area',
    'PCT_FEMALE': 'Percentage of female population in the geographic area',
    '4 year population change': 'Population change over the last 4 years',
    '4 year population change pct': 'Population change percentage over the last 4 years',
    'Mean Sale Price': 'Mean (average) sale price of homes sold in the area',
    'Mean Sale Price 5 yr Percent Change': '5-year percentage change in mean sale price',
    'Median Sale Price': 'Median sale price of homes sold in the area',
    'Median Sale Price 5 yr Percent Change': '5-year percentage change in median sale price',
    'Top Tier ZHVI': 'A measure of the typical home value and market changes across a given region and housing type. It reflects the typical value for homes in the 65th to 95th percentile range.',
    'Top Tier ZHVI 5 yr Percent Change': '5-year percentage change in top-tier ZHVI',
    'Mid Tier ZHVI': 'A measure of the typical home value and market changes across a given region and housing type. It reflects the typical value for homes in the 35th to 65th percentile range.',
    'Mid Tier ZHVI 5 yr Percent Change': '5-year percentage change in mid-tier ZHVI',
    'Bottom Tier ZHVI': 'A measure of the typical home value and market changes across a given region and housing type. It reflects the typical value for homes in the 5th to 35th percentile range.',
    'Bottom Tier ZHVI 5 yr Percent Change': '5-year percentage change in bottom-tier ZHVI',
    'Income Needed for Monthly Home Payment': 'An estimate of the annual household income required to spend less than 30% of monthly income on the total monthly payment after newly purchasing the typical home with a 20% down payment.',
    'Income Needed for Monthly Home Payment 5 yr Percent Change': '5-year percentage change in income needed for new home payments',
    'Income Needed for Monthly Rent Payment': 'An estimate of the household income required to spend less than 30% of monthly income to newly lease the typical rental.',
    'Income Needed for Monthly Rent Payment 5 yr Percent Change': '5-year percentage change in income needed for rent payments',
    'New Construction Sale Counts (Monthly)': 'Number of unique new construction homes sold monthly',
    'New Construction Sale Counts (Monthly) 5 yr Percent Change': '5-year percentage change in new construction sales',
    'Mean Days to Pending': 'Average number of days for homes to go from listed to pending status',
    'Mean Days to Pending 5 yr Percent Change': '5-year percentage change in days to pending',
    'Share of Listings with Price Cut': 'Percentage of listings that had price reductions during the month',
    'Share of Listings with Price Cut 5 yr Percent Change': '5-year percentage change in share of listings with price cuts',
    'Zillow Market Heat Index': 'The market heat index is a time series dataset that aims to capture the balance of for-sale supply and demand in a given market. A higher number means the market is more tilted in favor of sellers. It relies on a combination of engagement and listing performance inputs to provide insights into current market dynamics. It is calculated for single-family and condo homes.',
    'Zillow Market Heat Index 5 yr Percent Change': '5-year percentage change in market heat index',
    'Zillow Observed Rent Index (ZORI)': 'A smoothed measure of the typical observed market rate rent across a given region. ZORI is a repeat-rent index that is weighted to the rental housing stock to ensure representativeness across the entire market, not just those homes currently listed for-rent. The index is dollar-denominated by computing the mean of listed rents that fall into the 35th to 65th percentile range for all homes and apartments in a given region, which is weighted to reflect the rental housing stock.',
    'ZORI 5 yr Percent Change': '5-year percentage change in Zillow Observed Rent Index'
}

# Load data
@st.cache_data
def load_data():
    city_level = pd.read_csv('data/us_city_affordability_city_level.csv')
    msa_level = pd.read_csv('data/us_city_affordability.csv')
    
    # Trim MSA names to text before first dash
    for df in [city_level, msa_level]:
        if 'NAME' in df.columns:
            df['NAME'] = df['NAME'].str.split('-').str[0]
    
    # Add percentage columns for gender demographics
    for df in [city_level, msa_level]:
        if 'TOT_MALE' in df.columns and 'TOT_FEMALE' in df.columns and 'TOT_POP' in df.columns:
            # Calculate percentages (as decimals for consistency with other percent columns)
            df['PCT_MALE'] = (df['TOT_MALE'] / df['TOT_POP'])
            df['PCT_FEMALE'] = (df['TOT_FEMALE'] / df['TOT_POP'])
            
            # Round to 4 decimal places for precision
            df['PCT_MALE'] = df['PCT_MALE'].round(4)
            df['PCT_FEMALE'] = df['PCT_FEMALE'].round(4)
    
    return city_level, msa_level

city_level, msa_level = load_data()

# App title
st.title("📊 City Affordability Dashboard")

# Data level selection
st.markdown(f"<h4 style='color: {MCKINSEY_COLORS['primary']};'>Data Level Selection</h4>", unsafe_allow_html=True)
data_level = st.radio(
    "Choose Analysis Scope:",
    ["City Proper", "MSA (Metropolitan Statistical Area)"],
    horizontal=True
)

# Choose appropriate dataset
if data_level == "City Proper":
    df = city_level.copy()
else:
    df = msa_level.copy()



# Column selection for filtering
st.markdown("---")
st.markdown(f"<h4 style='color: {MCKINSEY_COLORS['primary']};'>Advanced Data Filtering</h4>", unsafe_allow_html=True)

# Get numeric columns for filtering
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if not numeric_cols:
    st.warning("No numeric columns available for filtering.")
else:
    # Multi-filter mode selection
    filter_mode = st.radio(
        "Filter Mode:",
        ["Single Column", "Multiple Columns"],
        horizontal=True,
        help="Choose between single or multiple column filtering"
    )
    
    if filter_mode == "Single Column":
        # Column selection
        filter_col = st.selectbox(
            "Select column to filter by:",
            options=numeric_cols,
            help="Choose a numeric column to create filter breaks"
        )
        selected_cols = [filter_col]
    else:
        # Multiple column selection
        selected_cols = st.multiselect(
            "Select columns to filter by:",
            options=numeric_cols,
            help="Choose multiple numeric columns for simultaneous filtering"
        )
    
    
    
    # Filter by selected columns
    if not selected_cols:
        st.warning("Please select at least one column to filter.")
        filtered_df = df.copy()
    else:
        filtered_df = df.copy()
        
        # Range slider filtering for each selected column
        st.markdown(f"<h5 style='color: {MCKINSEY_COLORS['primary']};'>Set Range Filters</h5>", unsafe_allow_html=True)
        
        for col in selected_cols:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            
            selected_range = st.slider(
                f"Select range for {col}:",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val),
                step=float((max_val - min_val) / 100),
                key=f"range_{col}"
            )
            
            filtered_df = filtered_df[
                (filtered_df[col] >= selected_range[0]) & 
                (filtered_df[col] <= selected_range[1])
            ]
                


# Display results
st.markdown("---")

if len(filtered_df) == 0:
    st.warning("No data matches the selected filters.")
else:
    # Create two-column layout: stats on left, data on right
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown(f"<h5 style='color: {MCKINSEY_COLORS['primary']};'>Statistical Summary</h5>", unsafe_allow_html=True)
        
        # Get numeric columns from filtered data
        numeric_filtered_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_filtered_cols:
            # Dropdown to select which variable to display
            selected_stat_col = st.selectbox(
                "Select variable for statistics:",
                options=numeric_filtered_cols,
                help="Choose which variable's statistics to display"
            )
            
            # Display statistics for selected column
            col_stats = filtered_df[selected_stat_col].describe()
            
            # Check if this is a percentage column to format appropriately
            is_pct = is_percentage_column(selected_stat_col)
            
            # Format function based on column type
            if is_pct:
                # For percentages, show as percentage with 2 decimals
                fmt = lambda x: f"{x*100:.2f}%"
            elif abs(col_stats['mean']) < 10:
                # For small numbers (likely already percentages or ratios), show decimals
                fmt = lambda x: f"{x:.3f}"
            else:
                # For large numbers, show as integers with commas
                fmt = lambda x: f"{int(x):,d}"
            
            st.markdown(f"""
            <div style="padding: 12px; background-color: {MCKINSEY_COLORS['light']}; border-radius: 6px; border-left: 4px solid {MCKINSEY_COLORS['secondary']}; margin-bottom: 10px;">
                <strong style="color: {MCKINSEY_COLORS['primary']}; font-size: 14px;">{selected_stat_col}</strong><br>
                <div style="font-size: 12px; color: #333333; margin-top: 5px;">
                    <div><strong style="color: {MCKINSEY_COLORS['primary']};">Count:</strong> <span style="color: #000000;">{int(col_stats['count']):,d}</span></div>
                    <div><strong style="color: {MCKINSEY_COLORS['primary']};">Mean:</strong> <span style="color: #000000;">{fmt(col_stats['mean'])}</span></div>
                    <div><strong style="color: {MCKINSEY_COLORS['primary']};">Median:</strong> <span style="color: #000000;">{fmt(col_stats['50%'])}</span></div>
                    <div><strong style="color: {MCKINSEY_COLORS['primary']};">Std Dev:</strong> <span style="color: #000000;">{fmt(col_stats['std'])}</span></div>
                    <div><strong style="color: {MCKINSEY_COLORS['primary']};">Min:</strong> <span style="color: #000000;">{fmt(col_stats['min'])}</span></div>
                    <div><strong style="color: {MCKINSEY_COLORS['primary']};">Max:</strong> <span style="color: #000000;">{fmt(col_stats['max'])}</span></div>
                    <div><strong style="color: {MCKINSEY_COLORS['primary']};">25th %:</strong> <span style="color: #000000;">{fmt(col_stats['25%'])}</span></div>
                    <div><strong style="color: {MCKINSEY_COLORS['primary']};">75th %:</strong> <span style="color: #000000;">{fmt(col_stats['75%'])}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No numeric columns available for statistical analysis.")
    
    with col2:
        st.markdown(f"<h5 style='color: {MCKINSEY_COLORS['primary']};'>Data Table</h5>", unsafe_allow_html=True)
        
        # Display options with reorderable columns
        all_columns = df.columns.tolist()  # Use original dataset columns, not filtered
        # Blacklist unwanted columns
        blacklisted_columns = ['Unnamed: 0']
        filtered_columns = [col for col in all_columns if col not in blacklisted_columns]
        
        display_options = st.multiselect(
            "Select columns to display:",
            options=filtered_columns,
            default=filtered_columns[:10] if len(filtered_columns) > 10 else filtered_columns
        )
        
        if display_options:
            # Show data table with formatted columns
            display_df = filtered_df[display_options].copy().reset_index(drop=True)
            
            # Format percentage columns for better readability
            for col in display_options:
                if is_percentage_column(col):
                    # Format as percentage with 2 decimals
                    display_df[col] = display_df[col].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "")
            
            st.dataframe(display_df)
            
            # Create scatterplot below data table
            st.markdown(f"<h5 style='color: {MCKINSEY_COLORS['primary']}; margin-top: 20px;'>Data Visualization</h5>", unsafe_allow_html=True)
            
            # Get numeric columns for plotting (excluding Name columns and blacklisted columns)
            numeric_cols_for_plot = []
            blacklisted_columns = ['Unnamed: 0']
            for col in filtered_df.columns:
                if (col.lower() not in ['name', 'city', 'region', 'state'] + [b.lower() for b in blacklisted_columns] and 
                    col not in blacklisted_columns and 
                    pd.api.types.is_numeric_dtype(filtered_df[col])):
                    numeric_cols_for_plot.append(col)
            
            if len(numeric_cols_for_plot) >= 2:
                # Plot controls section
                st.markdown(f"<h6 style='color: {MCKINSEY_COLORS['primary']};'>Plot Controls</h6>", unsafe_allow_html=True)
                
                # Column selectors for scatterplot
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Default to population growth if available, otherwise first numeric column
                    pop_cols = [col for col in numeric_cols_for_plot if 'population' in col.lower() and 'pct' in col.lower()]
                    if pop_cols:
                        x_col = st.selectbox("Select X-axis:", options=numeric_cols_for_plot, index=numeric_cols_for_plot.index(pop_cols[0]), key="x_axis")
                    else:
                        x_col = st.selectbox("Select X-axis:", options=numeric_cols_for_plot, key="x_axis")
                
                with col2:
                    # Default to ZHVI if available, otherwise second numeric column
                    zhvi_cols = [col for col in numeric_cols_for_plot if 'zhvi' in col.lower() and 'mid' in col.lower()]
                    if zhvi_cols:
                        y_col = st.selectbox("Select Y-axis:", options=[col for col in numeric_cols_for_plot if col != x_col], index=[col for col in numeric_cols_for_plot if col != x_col].index(zhvi_cols[0]), key="y_axis")
                    else:
                        y_col = st.selectbox("Select Y-axis:", options=[col for col in numeric_cols_for_plot if col != x_col], key="y_axis")
                
                with col3:
                    remaining_cols = [col for col in numeric_cols_for_plot if col not in [x_col, y_col]]
                    z_col = st.selectbox("Select Z-axis (optional):", options=["None"] + remaining_cols, key="z_axis")
                
                # Find name column for labels
                name_col = None
                for col in ['NAME', 'Name', 'City', 'Region', 'name', 'city', 'region']:
                    if col in filtered_df.columns:
                        name_col = col
                        break
                
                # Initialize outlier detection variables
                outlier_colors = None
                
                # Highlight outliers checkbox and info box
                col_check, col_info = st.columns([1, 2])
                
                with col_check:
                    highlight_outliers = st.checkbox("Highlight Outliers", value=False, key="highlight_outliers_plot")
                
                with col_info:
                    # Show outlier info box if highlighting is enabled
                    if highlight_outliers and outlier_colors is not None:
                        outlier_count = sum(1 for color in outlier_colors if color == MCKINSEY_COLORS['accent'])
                        st.markdown(f"""
                        <div style="padding: 15px; background-color: {MCKINSEY_COLORS['light']}; border-radius: 6px; border-left: 4px solid {MCKINSEY_COLORS['accent']}; margin-bottom: 20px;">
                            <strong style="color: {MCKINSEY_COLORS['primary']};">Highlight Outliers</strong><br>
                            <span style="color: {MCKINSEY_COLORS['text']};">
                            {outlier_count} outliers highlighted in McKinsey orange using Mahalanobis distance
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                
                if highlight_outliers:
                    try:
                        # Get numeric columns for Mahalanobis (use only plot variables)
                        plot_numeric_cols = [x_col, y_col]
                        if z_col != "None":
                            plot_numeric_cols.append(z_col)
                        
                        # Filter to only include complete rows
                        complete_data = plot_data[plot_numeric_cols].dropna()
                        
                        if len(complete_data) > 0 and len(plot_numeric_cols) >= 2:
                            X = complete_data[plot_numeric_cols].values
                            cov_matrix = np.cov(X, rowvar=False)
                            
                            # Check if covariance matrix is invertible
                            if np.linalg.det(cov_matrix) > 1e-10:
                                inv_cov_matrix = np.linalg.inv(cov_matrix)
                                mean_vector = np.mean(X, axis=0)
                                
                                # Calculate Mahalanobis distance
                                mahal_distances = []
                                for i in range(len(X)):
                                    mahal_dist = mahalanobis(X[i], mean_vector, inv_cov_matrix)
                                    mahal_distances.append(mahal_dist)
                                
                                # Calculate threshold
                                threshold = chi2.ppf(0.95, df=len(plot_numeric_cols))
                                
                                # Create outlier colors
                                complete_data = complete_data.copy()
                                complete_data['Is_Outlier'] = [dist > threshold for dist in mahal_distances]
                                
                                # Map back to original data
                                outlier_mask = plot_data.index.isin(complete_data[complete_data['Is_Outlier']].index)
                                outlier_colors = [MCKINSEY_COLORS['accent'] if outlier else MCKINSEY_COLORS['primary'] for outlier in outlier_mask]
                    except Exception as e:
                        st.warning(f"Could not calculate outliers: {str(e)}")
                        outlier_colors = MCKINSEY_COLORS['primary']
                
                
                
                # Create 2D or 3D plot based on selection
                if z_col == "None":
                    fig = px.scatter(
                        filtered_df, 
                        x=x_col, 
                        y=y_col,
                        title=f"{y_col} vs {x_col}",
                        color_discrete_sequence=[MCKINSEY_COLORS['primary']],
                        template='plotly_white',
                        text=filtered_df[name_col].astype(str) if name_col else None,
                        labels={
                            x_col: x_col,
                            y_col: y_col
                        }
                    )

                    fig.update_layout(
                        height=800,
                        autosize=False,
                        margin=dict(l=40, r=40, t=60, b=60)
                    )

                    # Now set formatting on the *axes*, not in labels:
                    # Use different formats depending on if it's a percentage column or not
                    x_is_pct = is_percentage_column(x_col)
                    y_is_pct = is_percentage_column(y_col)
                    
                    if x_is_pct:
                        fig.update_xaxes(tickformat=".1%")  # percentage format with 1 decimal (0.22 -> 22.0%)
                    else:
                        fig.update_xaxes(tickformat=",.0f")   # integers with commas
                    
                    if y_is_pct:
                        fig.update_yaxes(tickformat=".1%")  # percentage format with 1 decimal (0.22 -> 22.0%)
                    else:
                        fig.update_yaxes(tickformat=",.0f")   # integers with commas

                    
                    # Apply outlier colors if enabled
                    if highlight_outliers and outlier_colors is not None:
                        fig.update_traces(
                            marker=dict(
                                color=outlier_colors,
                                size=8,
                                opacity=0.7
                            ),
                            textposition='top center',
                            textfont=dict(size=10, color=MCKINSEY_COLORS['text']),
                            showlegend=False
                        )
                    else:
                        fig.update_traces(
                            marker=dict(
                                color=MCKINSEY_COLORS['primary'],
                                size=8,
                                opacity=0.7
                            ),
                            textposition='top center',
                            textfont=dict(size=10, color=MCKINSEY_COLORS['text']),
                            showlegend=False
                        )
                    
                else:
                    # 3D Scatterplot
                    fig = go.Figure(data=[go.Scatter3d(
                        x=filtered_df[x_col],
                        y=filtered_df[y_col],
                        z=filtered_df[z_col],
                        mode='markers+text',
                        text=filtered_df[name_col].astype(str) if name_col else None,
                        textposition='top center',
                        textfont=dict(size=10, color=MCKINSEY_COLORS['text']),
                        marker=dict(
                            size=6,
                            color=outlier_colors if highlight_outliers and outlier_colors is not None else MCKINSEY_COLORS['primary'],
                            opacity=0.7
                        )
                    )])
                    
                    fig.update_layout(
                        title=f"{z_col} vs {y_col} vs {x_col}",
                        title_font_color=MCKINSEY_COLORS['primary'],
                        scene=dict(
                            xaxis_title=x_col,
                            yaxis_title=y_col,
                            zaxis_title=z_col,
                            xaxis_title_font_color=MCKINSEY_COLORS['text'],
                            yaxis_title_font_color=MCKINSEY_COLORS['text'],
                            zaxis_title_font_color=MCKINSEY_COLORS['text'],
                            bgcolor='white'
                        ),
                        font_color=MCKINSEY_COLORS['text'],
                        paper_bgcolor=MCKINSEY_COLORS['light'],
                        height=800,
                        autosize=False,
                        margin=dict(l=40, r=40, t=60, b=60)
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add variable descriptions below the plot
                st.markdown("---")
                st.markdown(f"<h6 style='color: {MCKINSEY_COLORS['primary']};'>Selected Variable Descriptions</h6>", unsafe_allow_html=True)
                
                # Get the selected plot variables
                plot_variables = [x_col, y_col]
                if z_col != "None":
                    plot_variables.append(z_col)
                
                
                
                # Show descriptions for selected plot variables side-by-side
                desc_cols = st.columns(len(plot_variables))
                
                for i, var in enumerate(plot_variables):
                    with desc_cols[i]:
                        description = ZILLOW_VARIABLE_DESCRIPTIONS.get(var, "No description available for this variable.")
                        st.markdown(create_description_box(var, description), unsafe_allow_html=True)
                
            else:
                st.info("Scatterplot requires at least two numeric columns (excluding name fields).")
            
            # Download button
            csv = filtered_df[display_options].to_csv(index=False)
            st.download_button(
                label="Download filtered data as CSV",
                data=csv,
                file_name=f"filtered_{data_level.lower().replace(' ', '_')}_data.csv",
                mime="text/csv"
            )

# Footer with McKinsey styling
st.markdown("---")
st.markdown(f"""
<div style="padding: 20px; background-color: {MCKINSEY_COLORS['light']}; border-radius: 8px; border-left: 4px solid {MCKINSEY_COLORS['primary']}; text-align: center; margin-top: 30px;">
    <p style="color: {MCKINSEY_COLORS['text']}; margin: 0; font-size: 14px;">
        <strong style="color: {MCKINSEY_COLORS['primary']};">Data Sources:</strong> US Census Bureau & Zillow Housing Database
    </p>
    <p style="color: {MCKINSEY_COLORS['text']}; margin: 5px 0 0 0; font-size: 12px;">
        Last updated: 2024 | Analysis powered by advanced filtering algorithms
    </p>
</div>
""", unsafe_allow_html=True)