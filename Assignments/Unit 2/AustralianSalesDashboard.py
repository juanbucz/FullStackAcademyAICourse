# Import Libraries
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t
from scipy.stats import boxcox, yeojohnson, shapiro
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import streamlit as st

import platform
import sys



# ============================================
# Main Dashboard Landing Page  
# This has to be first streamlit command in order to display
# ============================================

st.set_page_config(
    page_title="Australian Apparel Sales Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# How to tell python version
#print (sys.version_info)
#print (platform.python_version())

# How to pip install from Terminal window:
# python -m pip install seaborn

# Put this at the top of your notebook
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)  # Set a large number
pd.set_option('display.max_colwidth', None)

# Set random seed for reproducibility
np.random.seed(315)

# ============================================
# Custom Style CSS
# ============================================
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }    
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================
# ============================================
# Utility Functions
# ============================================
# ============================================

# ============================================
# DATA LOADING & PREPROCESSING
# ============================================

# Cache Data Retrieval so we don't reload every refresh
@st.cache_data
def load_data():
    df = pd.read_csv('AusApparalSales4thQrt2020.csv')
    
    # Clean string columns
    df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)
    
    # Convert date
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Encode Date Features
    df['Month'] = df['Date'].dt.month
    df['MonthName'] = df['Date'].dt.month_name()
    df['Day'] = df['Date'].dt.day
    df['DayName'] = df['Date'].dt.day_name()
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['Quarter'] = df['Date'].dt.quarter

    df.rename(columns={'Time': 'TimeOfDay'}, inplace=True)

    # Data Type Conversions

    # int8 Conversions
    columns = ['Unit','Month','WeekOfYear']
    df[columns] = df[columns].astype('int8')

    columns = ['Day']
    df[columns] = df[columns].astype('int16')

    # int32 Conversions
    columns = ['Sales']
    df[columns] = df[columns].astype('int64')
    
    return df

def PrintDataFrameStatus(df_target):
    # Print Stats
    print("***********************************")
    print("Description Stats")
    print("***********************************")
    print()
    print(df_target.describe(include='all').T)
    print()

    # Print df Column Info
    print("***********************************")
    print("Basic Info of imported data set")
    print("***********************************")
    print()
    df_target.info()
    print()
    print()

    print(f'df_floridabikerentals Shape:{df_target.shape}')
    print()

    print('Do we have any features with null values?:')
    print(df_target.isnull().any().any())
    print()

    print('Feature Columns with that have null values:')
    print(df_target.isnull().sum()[df_target.isnull().sum() > 0])
    print()

    print('Do we have any features with nan values?:')
    print(df_target.isna().any().any())

    print("***********************************")
    print("First 20 rows of Data")
    print("***********************************")
    print()
    print(df_target.head(20))
    print()

    print("***********************************")
    print("First 20 rows of Random Sample Data")
    print("***********************************")
    print()
    print(df_target.sample(20))
    print

# Initial load of data
df_ausapparalsales = load_data()

def DisplayTable(df_target, new_fig_size=None, table_title=None):
    
    n_rows, n_cols = df_target.shape
    
    fig_size = new_fig_size
    if new_fig_size is None:
        fig_size = (n_cols * 1.5, (n_rows + 1) * 1.0)
        
    fig, ax = plt.subplots(fig_size)  # increased height multiplier
    ax.axis('off')

    # Format floats to 2 decimal places
    # I need help on this one for dynamic 
    cell_text = [[f'{val:.2f}' if isinstance(val, float) else str(val) for val in row] 
                 for row in df_target.values]

    table = ax.table(
        cellText=cell_text,
        colLabels=df_target.columns,
        #rowLabels=df_target.index,
        cellLoc='center',
        loc='center'
    )

    # Make column headers bold
    for col in range(n_cols):
        table[(0, col)].set_text_props(fontweight='bold')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(n_cols)))
    table.scale(1.2, 2.0)  # second param controls row height

    if table_title is not None:
        plt.title(table_title, fontweight='bold', fontsize=14)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close()

def DisplayNumericFeatureHistgorams(df_target, feature_columns = None, target_label=None, new_fig_size=None, plot_title=None, display_statistics=True):

    cols_to_plot = []
    statistics = []
    
    if feature_columns is None:
        cols_to_plot = df_target.select_dtypes(include=['number']).columns
    else:
        cols_to_plot = feature_columns

    if target_label is not None:
        cols_to_plot = cols_to_plot.drop(target_label).tolist()
    
    number_of_cols = len(cols_to_plot)

    ncols = min(number_of_cols, 3)
    nrows = (number_of_cols + ncols - 1) // ncols

    fig_size = new_fig_size
    if new_fig_size is None:
        fig_size = (5*ncols, 4*nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=fig_size)

    if number_of_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, col in enumerate(cols_to_plot):

        sns.histplot(data=df_target, x=col, ax=axes[i], kde=True)

        axes[i].set_xlabel(col, fontweight='bold')
        axes[i].set_ylabel('Count', fontweight='bold')
        data = df_target[col]

        if display_statistics == True:

            min_val = data.min()
            max_val = data.max()
            range_val = data.max() - data.min()
            mean_val = data.mean()
            median_val = data.median()
            mode_val = data.mode().tolist()
            std_val = data.std()
            quantile_val = data.quantile([0.25, 0.5, 0.75, 0.90, 0.95]).tolist()
            skew_val = data.skew()
            kurtosis_val = data.kurtosis()

            stats = {
                'Feature': col,
                'Min': min_val,
                'Max': max_val,
                'Mean': mean_val,
                'Median': median_val,
                'Mode': mode_val,
                'Std': std_val,
                'Range': range_val,
                'Quantile': quantile_val,
                'Skew': skew_val,
                'Kurtosis': kurtosis_val
            }            

            statistics.append(stats)
            axes[i].set_title(f'{col}\nmean={mean_val:.2f}, med={median_val:.2f}, std={data.std():.2f}', fontweight='bold')

            axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            axes[i].axvline(median_val, color='green', linestyle='-', linewidth=2, label=f'Median: {median_val:.2f}')            
            axes[i].axvspan(mean_val - std_val, mean_val + std_val, alpha=0.2, color='orange', label=f'±1 Std')

        else:
            axes[i].set_title(f'{col}', fontweight='bold')
        
        axes[i].tick_params(axis='both', labelsize=10)
        for label in axes[i].get_xticklabels() + axes[i].get_yticklabels():
            label.set_fontweight('bold')

    for i in range(number_of_cols, len(axes)):
        fig.delaxes(axes[i])

    title = None
    if plot_title is not None:
        plt.suptitle(title, y=1.02, fontsize=20, fontweight='bold')
    plt.subplots_adjust(top=0.92)  

    plt.tight_layout(h_pad=5)
    
    st.pyplot(fig)
    plt.close()

    # if display_statistics == True:
    #     df_statistics = pd.DataFrame(statistics)
    #     DisplayTable(df_statistics, 'Numeric Features Statistics')

def DisplayPieChart(df_target, feature_column, groupby_column, fig_size=None, plot_title=None, palette='viridis'):

    if fig_size is None:
        fig_size = (2.5, 2.5)

    fig, ax = plt.subplots(figsize=fig_size)
    time_sales = df_target.groupby(groupby_column)[feature_column].sum()

    # Convert palette string to list of colors
    if len(time_sales) > 0:
        colors = sns.color_palette(palette, len(time_sales))
        time_sales.plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=colors)

    ax.set_ylabel('')  # Remove the vertical "Sales" label

    if plot_title:
        plt.suptitle(plot_title, y=1.02, fontsize=12, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close() 

def DisplayBarChart(df_target, feature_column, groupby_column, fig_size=None, plot_title=None, horizontal=False, palette='viridis'):

    if fig_size is None:
        fig_size = (5, 4)

    fig, ax = plt.subplots(figsize=fig_size)
    
    data = df_target.groupby(groupby_column)[feature_column].sum().reset_index()
    
    if horizontal:
        sns.barplot(data=data, x=feature_column, y=groupby_column, ax=ax, palette=palette)
    else:
        sns.barplot(data=data, x=groupby_column, y=feature_column, ax=ax, palette=palette)
        plt.xticks(rotation=45)

    ax.set_xlabel(ax.get_xlabel(), fontweight='bold')
    ax.set_ylabel(ax.get_ylabel(), fontweight='bold')

    if plot_title:
        plt.suptitle(plot_title, y=1.02, fontsize=12, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def DisplayGroupedBarChart(df_target, feature_column, groupby_column, time_column, fig_size=None, plot_title=None, palette='viridis'):

    if fig_size is None:
        fig_size = (8, 5)

    fig, ax = plt.subplots(figsize=fig_size)
    
    data = df_target.groupby([groupby_column, time_column])[feature_column].sum().reset_index()

    sns.barplot(data=data, x=groupby_column, y=feature_column, hue=time_column, ax=ax, palette=palette)

    ax.set_xlabel(groupby_column, fontweight='bold')
    ax.set_ylabel(feature_column, fontweight='bold')
    plt.xticks(rotation=45)
    plt.legend(title=time_column)

    if plot_title:
        plt.suptitle(plot_title, y=1.02, fontsize=12, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def DisplayHeatmap(df_target, feature_column, groupby_column, time_column, fig_size=None, plot_title=None, palette='viridis'):

    if fig_size is None:
        fig_size = (8, 5)

    fig, ax = plt.subplots(figsize=fig_size)
    
    pivot = df_target.pivot_table(
        values=feature_column, 
        index=groupby_column, 
        columns=time_column, 
        aggfunc='sum'
    )

    sns.heatmap(pivot, annot=True, fmt='.0f', cmap=palette, ax=ax)

    ax.set_xlabel(time_column, fontweight='bold')
    ax.set_ylabel(groupby_column, fontweight='bold')

    if plot_title:
        plt.suptitle(plot_title, y=1.02, fontsize=12, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def DisplayLineChart(df_target, feature_column, groupby_column, time_column, fig_size=None, plot_title=None, palette='viridis'):

    if fig_size is None:
        fig_size = (8, 5)

    fig, ax = plt.subplots(figsize=fig_size)
    
    data = df_target.groupby([time_column, groupby_column])[feature_column].sum().reset_index()

    sns.lineplot(data=data, x=time_column, y=feature_column, hue=groupby_column, marker='o', ax=ax, palette=palette)

    ax.set_xlabel(time_column, fontweight='bold')
    ax.set_ylabel(feature_column, fontweight='bold')
    plt.xticks(rotation=45)
    plt.legend(title=groupby_column)

    if plot_title:
        plt.suptitle(plot_title, y=1.02, fontsize=12, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def CreateStatisticsSummary(df_target, group_cols):

    return df_target.groupby(group_cols).agg(
                                                Transactions=('Sales', 'count'),
                                                Total_Sales=('Sales', 'sum'),
                                                Avg_Sales=('Sales', 'mean'),
                                                Median_Sales=('Sales', 'median'),
                                                Avg_Units=('Unit', 'mean')
                                            ).round(2).reset_index()   


def DisplayStatisticsSummary(df_target, plot_title=None):

    if plot_title is not None:
        st.markdown(f'###### {plot_title}')

    st.dataframe(df_target.style.format({
        'Total_Sales': '${:,.0f}',
        'Avg_Sales': '${:,.0f}',
        'Median_Sales': '${:,.0f}',
        'Avg_Units': '{:.1f}'
    }), use_container_width=True)     

def format_currency(value):
    return f"${value:,.0f}"

def format_number(value):
    return f"{value:,.0f}"

# ============================================
# ============================================
# End Utility Functions
# ============================================
# ============================================



# ============================================
# Main Dashboard Page
# ============================================
st.header("🛍️ Australian Apparel Sales Dashboard")
st.markdown("---")

# ============================================
# Sidebar Reporting Filters
# ============================================    
all_states = df_ausapparalsales['State'].unique().tolist()
all_groups = df_ausapparalsales['Group'].unique().tolist()
all_times = df_ausapparalsales['TimeOfDay'].unique().tolist()

timeofday_mapping = {
                        "Daily": "DayName",
                        "Weekly": "WeekOfYear",
                        "Monthly": "MonthName",
                        "Quarterly": "Quarter"
                    }


# ============================================
# Sidebar Display
# ============================================    

st.sidebar.title("🎛️ Report Filters")
show_data_table = st.sidebar.checkbox("Show Raw Data", value=False)

# Emojis can be directly embedded into streamlit text!!
st.sidebar.header("🌏 Location")
selected_states = st.sidebar.multiselect(
    "Select States",
    options=all_states,
    default=all_states
)

st.sidebar.header("👥 Customer Segment")
selected_groups = st.sidebar.multiselect(
    "Select Groups",
    options=all_groups,
    default=all_groups
)

st.sidebar.header("🕐 Time of Day")
selected_times = st.sidebar.multiselect(
    "Select Time of Day",
    options=all_times,
    default=all_times
)
\

# ============================================
# FILTER DATA
# ============================================
df_ausapparalsales_filtered = df_ausapparalsales[
    (df_ausapparalsales['State'].isin(selected_states)) &
    (df_ausapparalsales['Group'].isin(selected_groups)) &
    (df_ausapparalsales['TimeOfDay'].isin(selected_times)) 
]


# ============================================
# Sales Summary Information
# ============================================    
st.markdown("#### 📈 Overall Sales Summary")
st.markdown("""
    <style>
    [data-testid="stMetricValue"] {
        font-size: 20px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

total_sales = df_ausapparalsales_filtered['Sales'].sum()
avg_sales = df_ausapparalsales_filtered['Sales'].mean()
total_transactions = len(df_ausapparalsales_filtered)
total_units = df_ausapparalsales_filtered['Unit'].sum()
avg_units = df_ausapparalsales_filtered['Unit'].mean()

col1.metric("💰 Total Sales", format_currency(total_sales))
col2.metric("📊 Avg Sale", format_currency(avg_sales))
col3.metric("🧾 Transactions", format_number(total_transactions))
col4.metric("📦 Total Units", format_number(total_units))
col5.metric("📦 Avg Units/Sale", f"{avg_units:.1f}")
st.markdown("---")

# ============================================
# Dashboard Reporting Tabs
# ============================================
tabOverview, tabState, tabGroup, tabTimeOfDay = st.tabs([
    "📊 Overview", 
    "🌏 State Analysis", 
    "👥 Group Analysis", 
    "📅 Time Of Day Analysis"
])

# ============================================
# Dashboard Reporting Tabs
# ============================================
with tabOverview:
    st.markdown("#### **Sales Overview**")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("###### Sales by State")

        DisplayBarChart(df_target=df_ausapparalsales_filtered, 
                        feature_column='Sales', 
                        groupby_column='State',
                        horizontal=True)

    with col2:
        st.markdown("###### Sales by Group")

        DisplayBarChart(df_target=df_ausapparalsales_filtered, 
                feature_column='Sales', 
                groupby_column='Group')


    col3, col4 = st.columns(2)

    with col3:
        st.markdown("###### Sales Distribution")

        DisplayNumericFeatureHistgorams(df_target=df_ausapparalsales_filtered, 
                                        feature_columns=['Sales'], 
                                        new_fig_size = (10,6), 
                                        display_statistics=True)

    with col4:    
        st.markdown("###### Sales by Time Of Day")

        DisplayPieChart(df_target=df_ausapparalsales_filtered, 
                        feature_column='Sales', 
                        groupby_column='TimeOfDay')



# ============================================
# Dashboard Reporting Tabs
# ============================================
with tabState:
    st.markdown("##### **State Sales Analysis**")

    report_type = st.selectbox("Select Report Type", 
                            ["Quarterly","Monthly","Weekly","Daily"],
                            key="state_time_select")
    
    df_state_summary = CreateStatisticsSummary(df_ausapparalsales_filtered, 'State')

    timeofday = timeofday_mapping[report_type]

    # Row 1: Charts (3 columns)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("###### Grouped Bar")
        DisplayGroupedBarChart(df_ausapparalsales_filtered, 'Sales', 'State', timeofday)

    with col2:
        st.markdown("###### Heatmap")
        DisplayHeatmap(df_ausapparalsales_filtered, 'Sales', 'State', timeofday)

    with col3:
        st.markdown("###### Trend Line")
        DisplayLineChart(df_ausapparalsales_filtered, 'Sales', 'State', timeofday)    

    # Row 2: Summary Table
    st.markdown("###### State Statistics Summary")
    DisplayStatisticsSummary(df_state_summary)

# ============================================
# Dashboard Reporting Tabs
# ============================================
with tabGroup:
    st.markdown("##### **Group Sales Analysis**")

    report_type = st.selectbox("Select Report Type", 
                            ["Quarterly","Monthly","Weekly","Daily"],
                            key="group_time_select")
    
    df_group_summary = CreateStatisticsSummary(df_ausapparalsales_filtered, 'Group')

    timeofday = timeofday_mapping[report_type]

    # Row 1: Charts (3 columns)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("###### Grouped Bar")
        DisplayGroupedBarChart(df_ausapparalsales_filtered, 'Sales', 'Group', timeofday)

    with col2:
        st.markdown("###### Heatmap")
        DisplayHeatmap(df_ausapparalsales_filtered, 'Sales', 'Group', timeofday)

    with col3:
        st.markdown("###### Trend Line")
        DisplayLineChart(df_ausapparalsales_filtered, 'Sales', 'Group', timeofday)    

    # Row 2: Summary Table
    st.markdown("###### Group Statistics Summary")
    DisplayStatisticsSummary(df_group_summary)    


# ============================================
# Dashboard Reporting Tabs
# ============================================
with tabTimeOfDay:
    st.markdown("##### **Time of Day Sales Analysis**")

    report_type = st.selectbox("Select Report Type", 
                            ["Quarterly","Monthly","Weekly","Daily"],
                            key="timeofday_time_select")

    df_timeofday_summary = CreateStatisticsSummary(df_ausapparalsales_filtered, 'TimeOfDay')

    timeofday = timeofday_mapping[report_type]

    # Row 1: Charts (3 columns)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("###### Grouped Bar")
        DisplayGroupedBarChart(df_ausapparalsales_filtered, 'Sales', 'TimeOfDay', timeofday)

    with col2:
        st.markdown("###### Heatmap")
        DisplayHeatmap(df_ausapparalsales_filtered, 'Sales', 'TimeOfDay', timeofday)

    with col3:
        st.markdown("###### Trend Line")
        DisplayLineChart(df_ausapparalsales_filtered, 'Sales', 'TimeOfDay', timeofday)    

    # Row 2: Summary Table
    st.markdown("###### Group Statistics Summary")
    DisplayStatisticsSummary(df_timeofday_summary)        