# Import necessary libraries
import streamlit as st  # For building the web app interface
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical computations
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # For evaluating model performance
import plotly.express as px  # For creating interactive visualizations
from statsmodels.tsa.holtwinters import ExponentialSmoothing  # For Exponential Smoothing model
from statsmodels.tsa.arima.model import ARIMA  # For ARIMA model
from prophet import Prophet  # For Prophet model
from prophet.plot import plot_plotly  # For plotting Prophet forecasts with Plotly
import matplotlib.pyplot as plt  # For additional plotting (if needed)
import warnings  # For handling warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set page configuration for the Streamlit app
st.set_page_config(
    layout="wide",  # Full width of the page
    page_title="Sales and Inventory Forecast App",  # Title of the app
    page_icon="📈"  # Favicon icon
)

#sdfsdfds
d#test
# Function to load the data
@st.cache_data(persist=True)
def load_data():
    """
    Load the sales data from a CSV file and parse the 'Order Date' column as datetime.
    Returns the loaded DataFrame.
    file_path = "category_ready_sales_data.csv"  # Adjust path to your dataset
    data = pd.read_csv(file_path, parse_dates=['Order Date'])  # Load data with 'Order Date' as datetime
    return data
    """
    
    file_path = "category_ready_sales_data.csv"  # Adjust path to your dataset
    data = pd.read_csv(file_path)  # Load data without initial date parsing
    data['Order Date'] = pd.to_datetime(data['Order Date'], format='%d/%m/%Y', errors='coerce')  # For some reason only the explicit conversion works.
    return data

# Function to dynamically aggregate data based on user selections
def aggregate_data(data, granularity, category_filters):
    """
    Aggregate the sales data based on the selected granularity and categories.

    Parameters:
    - data: The original sales DataFrame.
    - granularity: The selected time granularity for aggregation (e.g., 'Daily', 'Weekly').
    - category_filters: The list of categories selected by the user.

    Returns:
    - aggregated: The aggregated sales DataFrame.
    """
    # Filter data to include only selected categories
    filtered_data = data[data['Category'].isin(category_filters)]
    filtered_data = filtered_data.set_index('Order Date')  # Set 'Order Date' as the index for resampling

    # Resample data based on selected granularity
    if granularity == "Daily":
        aggregated = filtered_data.resample('D').sum()
    elif granularity == "Weekly":
        aggregated = filtered_data.resample('W').sum()
    elif granularity == "Monthly":
        aggregated = filtered_data.resample('M').sum()
    elif granularity == "Quarterly":
        aggregated = filtered_data.resample('Q').sum()
    else:
        aggregated = filtered_data.resample('D').sum()  # Default to Daily if unknown
    return aggregated

# Function to convert DataFrame to CSV for download
@st.cache_data
def convert_df(df):
    """
    Convert a DataFrame to a CSV format and encode it in UTF-8.

    Parameters:
    - df: The DataFrame to convert.

    Returns:
    - Encoded CSV data ready for download.
    """
    return df.to_csv(index=True).encode('utf-8')

# Main function to run the Streamlit app
def main():
    # Sidebar options for user input
    st.sidebar.title("Settings")
    st.sidebar.subheader("Category and Aggregation")
    df = load_data()  # Load the dataset

    # Multiselect widget for category selection
    category_filter = st.sidebar.multiselect(
        "Select Category",
        options=df['Category'].unique(),
        default=df['Category'].unique(),  # Default to all categories selected
        help="Choose one or more product categories to include in the analysis."
    )

    # Dropdown menu for selecting aggregation level
    aggregation = st.sidebar.selectbox(
        "Choose Aggregation Level",
        ["Daily", "Weekly", "Monthly", "Quarterly"],
        index=2,  # Set 'Monthly' as the default option (0-based index)
        help="Select the time interval over which to aggregate the sales data."
    )

    # Forecast Horizon Slider repositioned and range reduced to 60
    forecast_steps = st.sidebar.slider(
        "Forecast Horizon (steps)",
        min_value=1,
        max_value=60,  # Reduced maximum value from 365 to 60
        value=12,  # Set default value to 12
        help="Select the number of future time steps to forecast."
    )

    # Aggregate data based on user selections
    aggregated_data = aggregate_data(df, aggregation, category_filter)

    # Model selection
    st.sidebar.subheader("Model Selection")
    model_name = st.sidebar.selectbox(
        "Choose Forecasting Model",
        ("Prophet", "Exponential Smoothing", "ARIMA"),  # Prophet is now the first option
        help="Select the forecasting model to use."
    )

    # Dynamic model-specific configuration based on the selected model
    if model_name == "Prophet":
        st.sidebar.subheader("Prophet Configuration")
        prophet_seasonality = st.sidebar.selectbox(
            "Seasonality Mode",
            ["additive", "multiplicative"],
            help="Choose the type of seasonality for the model."
        )
        changepoint_prior_scale = st.sidebar.slider(
            "Changepoint Prior Scale",
            0.001, 0.5,
            value=0.05,
            help="Adjust the flexibility of the trend by changing the scale of the changepoints."
        )
        seasonality_prior_scale = st.sidebar.slider(
            "Seasonality Prior Scale",
            0.01, 10.0,
            value=10.0,
            help="Adjust the strength of the seasonality component."
        )
    elif model_name == "Exponential Smoothing":
        st.sidebar.subheader("Exponential Smoothing Configuration")
        trend = st.sidebar.selectbox(
            "Trend Type",
            ["add", "mul", None],
            help="Select the type of trend component."
        )
        seasonal = st.sidebar.selectbox(
            "Seasonality Type",
            ["add", "mul", None],
            help="Select the type of seasonal component."
        )
        seasonal_periods = st.sidebar.slider(
            "Seasonal Periods",
            1, 365,
            value=12,
            help="Specify the number of periods in a complete seasonal cycle."
        )
    elif model_name == "ARIMA":
        st.sidebar.subheader("ARIMA Configuration")
        arima_p = st.sidebar.slider(
            "AR Order (p)",
            0, 5,
            value=1,
            help="Set the number of lag observations included in the model (autoregressive terms)."
        )
        arima_d = st.sidebar.slider(
            "Differencing (d)",
            0, 2,
            value=1,
            help="Set the number of times that the raw observations are differenced."
        )
        arima_q = st.sidebar.slider(
            "MA Order (q)",
            0, 5,
            value=1,
            help="Set the size of the moving average window."
        )

    # Tabs for different sections of the app
    tab1, tab2, tab3 = st.tabs(["Data Overview", "Descriptive Analytics", "Forecasting"])

    with tab1:
        # Display the raw dataset
        st.subheader("Raw Dataset")
        st.dataframe(df, height=300)  # Scrollable dataframe for raw data

        # Display the aggregated data
        st.subheader(f"Aggregated Data ({aggregation} Aggregation) for Selected Categories")
        st.dataframe(aggregated_data, height=300)  # Scrollable dataframe for aggregated data

        # Download button for aggregated data
        csv_aggregated = convert_df(aggregated_data)
        st.download_button(
            label="Download Aggregated Data as CSV",
            data=csv_aggregated,
            file_name='aggregated_data.csv',
            mime='text/csv',
            key='download-aggregated',
            help="Download the aggregated data for your own analysis."
        )

    with tab2:
        # Descriptive analytics section
        st.subheader("Descriptive Analytics")
        st.write("Basic statistics of the sales data:")
        st.write(aggregated_data.describe())  # Display statistical summary

        # Interactive line chart showing sales over time
        fig = px.line(
            aggregated_data,
            x=aggregated_data.index,
            y='Sales',
            title='Sales Over Time',
            labels={'Sales': 'Sales', 'index': 'Date'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Forecasting section
        st.subheader("Forecasting")
        if st.button("Run Forecast", help="Click to run the forecast using the selected model and settings."):
            try:
                with st.spinner('Running Forecast...'):
                    # Loop through each selected category
                    for category in category_filter:
                        st.markdown(f"### Forecast for **{category}**")

                        # Filter and aggregate data for the current category
                        category_data = df[df['Category'] == category].set_index('Order Date')
                        aggregated = aggregate_data(df, aggregation, [category])

                        if aggregated.empty:
                            st.warning(f"No data available for category: {category}")
                            continue

                        # Ensure the index is sorted chronologically
                        aggregated = aggregated.sort_index()

                        # Split data into training and testing sets (80% training, 20% testing)
                        split_index = int(len(aggregated) * 0.8)
                        train_data = aggregated.iloc[:split_index]
                        test_data = aggregated.iloc[split_index:]

                        # Define frequency for date range generation
                        freq_map = {
                            "Daily": 'D',
                            "Weekly": 'W',
                            "Monthly": 'M',
                            "Quarterly": 'Q'
                        }
                        freq = freq_map.get(aggregation, 'D')

                        # Model fitting and forecasting
                        if model_name == "Prophet":
                            # Prepare training data for Prophet
                            train_prophet_df = train_data.reset_index().rename(columns={'Order Date': 'ds', 'Sales': 'y'})

                            # Initialize and fit the Prophet model on training data
                            model = Prophet(
                                seasonality_mode=prophet_seasonality,
                                changepoint_prior_scale=changepoint_prior_scale,
                                seasonality_prior_scale=seasonality_prior_scale
                            )
                            model.fit(train_prophet_df)

                            # Create a future dataframe including all dates from historical data and future dates
                            total_periods = len(aggregated) + forecast_steps
                            future_dates = pd.date_range(start=aggregated.index[0], periods=total_periods, freq=freq)
                            future = pd.DataFrame({'ds': future_dates})

                            # Predict on future dates (includes training, test data dates, and future dates)
                            forecast = model.predict(future)

                            # Extract predictions for the test set
                            test_forecast = forecast[forecast['ds'].isin(test_data.index)]['yhat'].values

                            # Generate future forecast values
                            future_forecast = forecast.iloc[-forecast_steps:][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

                        elif model_name == "Exponential Smoothing":
                            # Fit the Exponential Smoothing model
                            model = ExponentialSmoothing(
                                train_data['Sales'],
                                trend=trend,
                                seasonal=seasonal,
                                seasonal_periods=seasonal_periods
                            )
                            fitted_model = model.fit()
                            test_forecast = fitted_model.forecast(steps=len(test_data))

                            # Fit the model on the full dataset for future forecasting
                            full_model = ExponentialSmoothing(
                                aggregated['Sales'],
                                trend=trend,
                                seasonal=seasonal,
                                seasonal_periods=seasonal_periods
                            )
                            full_fitted_model = full_model.fit()
                            future_forecast = full_fitted_model.forecast(steps=forecast_steps)
                        elif model_name == "ARIMA":
                            # Fit the ARIMA model
                            model = ARIMA(train_data['Sales'], order=(arima_p, arima_d, arima_q))
                            fitted_model = model.fit()
                            test_forecast = fitted_model.forecast(steps=len(test_data))

                            # Fit the model on the full dataset for future forecasting
                            full_model = ARIMA(aggregated['Sales'], order=(arima_p, arima_d, arima_q))
                            full_fitted_model = full_model.fit()
                            future_forecast = full_fitted_model.forecast(steps=forecast_steps)

                        # Plotting the forecast results
                        if model_name != "Prophet":
                            # Generate dates for the forecasted periods
                            last_date = aggregated.index[-1]
                            forecast_dates = pd.date_range(last_date + pd.Timedelta(1, unit='D'), periods=forecast_steps, freq=freq)
                            forecast_series = pd.Series(future_forecast, index=forecast_dates)

                            # Create an interactive Plotly figure
                            fig_forecast = px.line(
                                aggregated['Sales'],
                                title=f"{model_name} Forecast for {category}",
                                labels={'value': 'Sales', 'index': 'Date'}
                            )
                            fig_forecast.add_scatter(
                                x=forecast_series.index,
                                y=forecast_series.values,
                                mode='lines',
                                name='Future Forecast'
                            )
                            st.plotly_chart(fig_forecast, use_container_width=True)
                        else:
                            # Use Prophet's built-in plotting function
                            # Plot the model's predictions over the entire period
                            fig_forecast = plot_plotly(model, forecast)
                            fig_forecast.update_layout(
                                title=f"{model_name} Forecast for {category}",
                                xaxis_title='Date',
                                yaxis_title='Sales'
                            )
                            st.plotly_chart(fig_forecast, use_container_width=True)

                        # Model evaluation metrics
                        st.subheader("Model Evaluation")
                        if len(test_forecast) == 0:
                            st.warning("No predictions available for evaluation.")
                        else:
                            # Calculate performance metrics
                            mae = mean_absolute_error(test_data['Sales'], test_forecast)
                            mse = mean_squared_error(test_data['Sales'], test_forecast)
                            rmse = np.sqrt(mse)
                            r2 = r2_score(test_data['Sales'], test_forecast)

                            # Create a DataFrame to display metrics
                            metrics = {
                                "Metric": ["R-squared (R²)", "Mean Absolute Error (MAE)",
                                           "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)"],
                                "Value": [r2, mae, mse, rmse]
                            }
                            metrics_df = pd.DataFrame(metrics)

                            # Display the metrics table without row numbers
                            st.table(metrics_df.reset_index(drop=True))

                            # Plot actual vs predicted sales for the test period
                            comparison_df = pd.DataFrame({
                                "Actual Sales": test_data['Sales'],
                                "Predicted Sales": test_forecast
                            }, index=test_data.index)

                            fig_comparison = px.line(
                                comparison_df,
                                x=comparison_df.index,
                                y=["Actual Sales", "Predicted Sales"],
                                title="Actual vs Predicted Sales",
                                labels={'value': 'Sales', 'index': 'Date'}
                            )
                            st.plotly_chart(fig_comparison, use_container_width=True)

                        # Prepare forecast data for download
                        if model_name != "Prophet":
                            # For ARIMA and Exponential Smoothing models
                            forecast_df = pd.DataFrame({
                                "Date": forecast_dates,
                                "Forecasted Sales": future_forecast
                            })
                            forecast_df["Actual Sales"] = np.nan  # No actual sales data for future dates

                            # Include actual vs predicted for the test period
                            test_forecast_df = pd.DataFrame({
                                "Date": test_data.index,
                                "Actual Sales": test_data['Sales'],
                                "Predicted Sales": test_forecast
                            })
                            combined_forecast_df = pd.concat([test_forecast_df, forecast_df], ignore_index=True)
                        else:
                            # For Prophet model
                            if len(test_forecast) > 0:
                                # Future forecast data
                                future_forecast_df = future_forecast.rename(columns={'ds': 'Date', 'yhat': 'Forecasted Sales'})
                                future_forecast_df["Actual Sales"] = np.nan

                                # Filter the forecast to include only historical dates
                                historical_forecast = forecast[forecast['ds'].isin(aggregated.index)][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                                historical_forecast = historical_forecast.rename(columns={'ds': 'Date', 'yhat': 'Predicted Sales'})

                                # Assign actual sales values
                                historical_forecast['Actual Sales'] = aggregated['Sales'].values

                                # Combine historical predictions and future forecasts
                                combined_forecast_df = pd.concat([
                                    historical_forecast[['Date', 'Actual Sales', 'Predicted Sales']],
                                    future_forecast_df[['Date', 'Forecasted Sales', 'Actual Sales']]
                                ], ignore_index=True)
                            else:
                                st.warning("No predictions available for Prophet to download.")
                                combined_forecast_df = pd.DataFrame()

                        # Set 'Date' as the index for the combined DataFrame
                        if not combined_forecast_df.empty:
                            combined_forecast_df.set_index("Date", inplace=True)

                            # Provide download button for forecast results
                            csv_forecast = convert_df(combined_forecast_df)
                            st.download_button(
                                label=f"Download Forecast for {category} as CSV",
                                data=csv_forecast,
                                file_name=f'forecast_{category}.csv',
                                mime='text/csv',
                                key=f'download-forecast-{category}',
                                help="Download the forecast results as a CSV file."
                            )
                        else:
                            st.info("No forecast data available for download.")

            except Exception as e:
                st.error(f"Error during {model_name} forecasting: {e}")

# Run the app when the script is executed
if __name__ == "__main__":
    main()
