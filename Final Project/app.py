import datetime
import streamlit as st, pandas as pd, numpy as np
from datetime import date, timedelta
import yfinance as yf
import requests
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
#from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
from scipy import stats
st.title('Financial Dashboard 📈')
st.image('stock_market.jpg')


ticker=st.sidebar.text_input('Name of Stock',value="AAPL")
# Function to convert selected duration to start and end dates
def get_start_end_dates(selected_duration):
    end_date = datetime.date.today()
    if selected_duration == "1 Year":
        start_date = end_date - datetime.timedelta(days=365)
    elif selected_duration == "5 Years":
        start_date = end_date - datetime.timedelta(days=5*365)
    elif selected_duration == "10 Years":
        start_date = end_date - datetime.timedelta(days=10*365)
    else:
        # Default to 1 year
        start_date = end_date - datetime.timedelta(days=365)
    return start_date, end_date

# Sidebar inputs
#ticker = st.sidebar.text_input('Name of Stock', value="AAPL")
selected_duration = st.sidebar.selectbox('Select Duration', ["1 Year", "5 Years", "10 Years"])

# Convert selected duration to start and end dates
start_date, end_date = get_start_end_dates(selected_duration)

# Output selected start and end dates
# st.sidebar.write(f"Start Date: {start_date}")
# st.sidebar.write(f"End Date: {end_date}")
# start_date= st.sidebar.date_input('Start Date',datetime.date(2024, 1, 1))
# end_date= st.sidebar.date_input('End Date')
data=yf.download(ticker,start=start_date,end=end_date)
api_key = 'kR3pOO0UkPTyu5b4gq6pYCmk4tk8mga6'
requestString = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={api_key}"
request = requests.get(requestString)
json_data = request.json()
data1 = json_data[0]
#st.write(data1)


# Calculate the start and end dates for the last 10 years
#end_date = datetime.today()
start_date = end_date - timedelta(days=365.25*10)

# Download the monthly stock data for the last 10 years
stock_data = yf.download(ticker, start=start_date, end=end_date, interval="1mo")
market_data = yf.download('^GSPC', start=start_date, end=end_date, interval="1mo")
Introduction, Financial_Ratio, Monte_Carlo_Simulation, Predictions, GoldenCross = st.tabs(["Introduction","Financial Ratio","Monte Carlo Simulation", "Predictions", "Golden Cross"])
 
with Introduction:
    # with st.write("About Company"):
    st.subheader("About Company")
    st.write(data1["description"])
    st.subheader("Stock Data")
    fig=px.line(data,x=data.index,y=data['Adj Close'],title= ticker)
    st.plotly_chart(fig)
    with st.expander("Financial Data"):
        data2 = data.copy()
        data2['% Change'] = data['Adj Close'].shift(1) - 1
        data2.dropna(inplace=True)
        st.write(data2)

with Financial_Ratio:
    # Balance Sheet
    with st.expander("Financial Statement"):
        balance_sheet_url = f'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?apikey={api_key}'
        balance_sheet_request = requests.get(balance_sheet_url)
        balance_sheet_data = balance_sheet_request.json()
        bs = pd.DataFrame(balance_sheet_data)
        bs.set_index('date', inplace=True)
        st.subheader('Balance Sheet')
        st.write(bs.T)
    # Income Statement
        income_statement_url = f'https://financialmodelingprep.com/api/v3/income-statement/{ticker}?apikey={api_key}'
        income_statement_request = requests.get(income_statement_url)
        income_statement_data = income_statement_request.json()
        is1 = pd.DataFrame(income_statement_data)
        is1.set_index('date', inplace=True)
        st.subheader('Income Statement')
        st.write(is1.T)
    # Cash Flow Statement
        cash_flow_url = f'https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}?apikey={api_key}'
        cash_flow_request = requests.get(cash_flow_url)
        cash_flow_data = cash_flow_request.json()
        cf = pd.DataFrame(cash_flow_data)
        cf.set_index('date', inplace=True)
        st.subheader('Cash Flow Statement')
        st.write(cf.T)


# Check if bs contains data
    st.subheader("Liquidity Ratios")    
    if not bs.empty:
    # Calculate ratios
        current_ratio = round(bs.iloc[0]['totalCurrentAssets'] / bs.iloc[0]['totalCurrentLiabilities'], 2)
        quick_ratio = round((bs.iloc[0]['totalCurrentAssets'] - bs.iloc[0]['inventory']) / bs.iloc[0]['totalCurrentLiabilities'], 2)
        cash_ratio = round(bs.iloc[0]['cashAndCashEquivalents'] / bs.iloc[0]['totalCurrentLiabilities'], 2)

    # Define formulas for each ratio
        current_formula = "Total Current Assets / Total Current Liabilities"
        quick_formula = "(Total Current Assets - Inventory) / Total Current Liabilities"
        cash_formula = "Cash and Cash Equivalents / Total Current Liabilities"

    # Create a DataFrame for the table
        table_data = pd.DataFrame({
            "Metric": ["Current Ratio", "Quick Ratio", "Cash Ratio"],
            "Value": [current_ratio, quick_ratio, cash_ratio],
            "Formula": [current_formula, quick_formula, cash_formula]
        })

    # Display the table without index numbers
        st.dataframe(table_data)
    else:
        st.write("Balance sheet data is empty. Please check your data.")

   
    # Assuming is1, bs, and cf are dictionaries containing income statement, balance sheet, and cash flow statement data respectively
    #st.write(data1)
    st.subheader("Profitability Ratios")    
    if not is1.empty:
        if not bs.empty:
            if not cf.empty:
            # Calculate ratios
                current_ratio = round(bs.iloc[0]['totalCurrentAssets'] / bs.iloc[0]['totalCurrentLiabilities'], 2)
                quick_ratio = round((bs.iloc[0]['totalCurrentAssets'] - bs.iloc[0]['inventory']) / bs.iloc[0]['totalCurrentLiabilities'], 2)
                cash_ratio = round(bs.iloc[0]['cashAndCashEquivalents'] / bs.iloc[0]['totalCurrentLiabilities'], 2)
                gpm = round((is1.iloc[0]['grossProfit'] / is1.iloc[0]['revenue']) * 100, 2)
                opm = round((is1.iloc[0]['operatingIncome'] / is1.iloc[0]['revenue']) * 100, 2)
                roa = round((is1.iloc[0]['netIncome'] / bs.iloc[0]['totalAssets']) * 100, 2)
                roe = round((is1.iloc[0]['netIncome'] - cf.iloc[0]['dividendsPaid']) / bs.iloc[0]['totalStockholdersEquity'] * 100, 2)
                ros = round((is1.iloc[0]['operatingIncome'] - is1.iloc[0]['operatingExpenses']) / is1.iloc[0]['revenue'] * 100, 2)
                roi = round((is1.iloc[0]['netIncome'] / is1.iloc[0]['costOfRevenue']) * 100, 2)

            # Define formulas for each ratio
                gpm_formula = "(Gross Profit / Revenue) * 100"
                opm_formula = "(Operating Income / Revenue) * 100"
                roa_formula = "(Net Income / Total Assets) * 100"
                roe_formula = "(Net Income - Dividends) / Total Stockholders Equity) * 100"
                ros_formula = "((Operating Income - Operating Expenses) / Revenue) * 100"
                roi_formula = "(Net Income / Cost of Revenue) * 100"


            # Create a DataFrame for the table
                table_data = pd.DataFrame({
                    "Metric": ["Gross Profit Margin (%)", "Operating Profit Margin (%)", "Return on Assets (%)", "Return on Equity (%)", "Return on Sales (%)", "Return on Investment (%)"],
                    "Value": [gpm, opm, roa, roe, ros, roi],
                    "Formula": [gpm_formula, opm_formula, roa_formula, roe_formula, ros_formula, roi_formula]
                })

            # Display the table without index numbers
                st.dataframe(table_data)
            else:
                st.write("Income statement data is empty. Please check your data.")

# Assuming is1, bs, cf, and company_info are dictionaries containing data for income statement, balance sheet, cash flow statement, and company information respectively

    st.subheader("Earning Ratio")

    # Check if income statement (is1) contains data
    if not is1.empty:
        # Check if balance sheet (bs) and cash flow statement (cf) contain data
        if not bs.empty and not cf.empty:
            # Calculate ratios
            pe_ratio = round(data1['price'] / is1.iloc[0]['eps'], 2)
            dividend_payout_ratio = round(cf.iloc[0]['dividendsPaid'] / is1.iloc[0]['netIncome'], 2)
            debt_to_equity_ratio = round(bs.iloc[0]['netDebt'] / bs.iloc[0]['totalStockholdersEquity'], 2)
            sustainable_growth_rate = roe * (1 - dividend_payout_ratio)

            # Define formulas for each ratio
            pe_ratio_formula = "Price / Earnings per Share"
            dividend_payout_ratio_formula = "(Dividends Paid / Net Income)"
            debt_to_equity_ratio_formula = "(Net Debt / Total Stockholders Equity)"
            sustainable_growth_rate_formula = "(ROE * (1 - Dividend Payout Ratio))"

            # Create a DataFrame for the table
            table_data = pd.DataFrame({
                "Metric": ["Price-to-Earnings Ratio", "Dividend Payout Ratio",
                        "Debt-to-Equity Ratio", "Sustainable Growth Rate"],
                "Value": [pe_ratio, dividend_payout_ratio, debt_to_equity_ratio, sustainable_growth_rate],
                "Formula": [pe_ratio_formula, dividend_payout_ratio_formula,
                            debt_to_equity_ratio_formula, sustainable_growth_rate_formula]
            })

            # Display the table without index numbers
            st.dataframe(table_data)
        else:
            st.write("Balance sheet or cash flow statement data is empty. Please check your data.")
    


    
    st.subheader("CAPM and WACC")

    # Ensure is1 DataFrame is not empty
    if not is1.empty:
        market_data['Returns'] = market_data['Adj Close'].pct_change()
        stock_data['Returns'] = stock_data['Adj Close'].pct_change()
        market_returns = market_data['Returns'].values[1:]
        stock_returns = stock_data['Returns'].values[1:]

        beta = round(stats.linregress(market_returns, stock_returns)[0], 4)

        # Canada Long Term Real Return Bonds Rate - https://ycharts.com/indicators/canada_long_term_real_return_bonds_rate
        rf = 1.74
        gm = stats.gmean(1 + market_returns) - 1
        period = 12
        nominal_rate = gm * period
        rm = (((1 + (nominal_rate / period)) ** period) - 1) * 100

        # CAPM
        cost_of_equity = round(rf + beta * (rm - rf), 4)

        # Check if 'totalEquity' key exists in bs DataFrame
        if 'totalEquity' in bs.columns:
            equity = bs.iloc[0]['totalEquity']
            debt = bs.iloc[0]['totalDebt']

            # Total Funding
            total_funding = equity + debt

            # Check if 'interestExpense' key exists in is1 DataFrame
            if 'interestExpense' in is1.columns:
                cost_of_debt = round(is1.iloc[0]['interestExpense'] / bs.iloc[0]['totalDebt'], 4) * 100
                
                # Check if 'incomeTaxExpense' and 'incomeBeforeTax' keys exist in is1 DataFrame
                if 'incomeTaxExpense' in is1.columns and 'incomeBeforeTax' in is1.columns:
                    # Calculate income tax rate
                    income_tax_rate = round(is1.iloc[0]['incomeTaxExpense'] / is1.iloc[0]['incomeBeforeTax'], 4)

                    # Define the calculate_wacc function
                    def calculate_wacc(market_value_of_equity, market_value_of_debt, cost_of_equity, cost_of_debt, corporate_tax_rate):
                        total_value = market_value_of_equity + market_value_of_debt
                        equity_ratio = market_value_of_equity / total_value
                        debt_ratio = market_value_of_debt / total_value
                        
                        wacc = (equity_ratio * cost_of_equity) + (debt_ratio * cost_of_debt * (1 - corporate_tax_rate))
                        return wacc

                    # Calculate WACC
                    wacc_value = calculate_wacc(market_value_of_equity=equity,
                                                market_value_of_debt=debt,
                                                cost_of_equity=cost_of_equity,
                                                cost_of_debt=cost_of_debt,
                                                corporate_tax_rate=income_tax_rate)

                    # Create DataFrame for CAPM and WACC
                    table_data = pd.DataFrame({
                        "Metric": ["CAPM", "WACC"],
                        "Value": [cost_of_equity, wacc_value],
                        "Formula": ["Risk-free Rate + Beta * (Market Return - Risk-free Rate)",
                                    "(Equity Ratio * Cost of Equity) + (Debt Ratio * Cost of Debt * (1 - Corporate Tax Rate))"]
                    })

                    # Display the table without index numbers
                    st.table(table_data)

                else:
                    st.write("Error: 'incomeTaxExpense' or 'incomeBeforeTax' not found in DataFrame.")
            else:
                st.write("Error: 'interestExpense' not found in DataFrame.")
        else:
            st.write("Error: 'totalEquity' not found in DataFrame.")
    else:
        st.write("Error: DataFrame 'is1' is empty.")


    
    


with Monte_Carlo_Simulation:
    def fetch_stock_data(ticker, start_date, end_date):
        data = yf.download(ticker, start=start_date, end=end_date)
        return data

    # Function to perform Monte Carlo simulation
    def monte_carlo_simulation(data, n_simulations, n_days):
        last_price = data['Adj Close'][-1]
        simulation_df = pd.DataFrame()
        
        for x in range(n_simulations):
            count = 0
            daily_volatility = data['Returns'].std()
            price_series = []
            
            # Generating the price list for the next year
            price = last_price * (1 + np.random.normal(0, daily_volatility))
            price_series.append(price)
            
            for y in range(n_days):
                if count == n_days+1:
                    break
                price = price_series[count] * (1 + np.random.normal(0, daily_volatility))
                price_series.append(price)
                count += 1
            
            simulation_df[x] = price_series
        
        return simulation_df

    # Main Streamlit app
    def main():
        st.header('Stock Price Prediction with Monte Carlo Simulation')
        
        # Input text boxes for user input
        #ticker = st.text_input('Enter Stock Ticker (e.g., AAPL for Apple)', 'AAPL')
        n_simulations = st.number_input('Number of Simulations', min_value=1, max_value=1000, value=100, step=1)
        n_days = st.number_input('Number of Days to Project', min_value=1, max_value=365, value=30, step=1)
        
        # Fetching data
        #end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        data = fetch_stock_data(ticker, start_date, end_date)
        
        # Plotting last year's stock price chart
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=data.index, y=data['Adj Close'], mode='lines', name='Stock Price'))
        fig1.update_layout(title=f'Last Year Stock Price Chart of {ticker}', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig1)
        
        # Calculating returns and performing Monte Carlo simulation
        data['Returns'] = data['Adj Close'].pct_change()
        simulation_df = monte_carlo_simulation(data, n_simulations, n_days)
        
        # Plotting Monte Carlo simulation
            # Plotting Monte Carlo simulation
        fig2 = go.Figure()
        for col in simulation_df.columns:
            x_values = list(range(n_days+1))
            fig2.add_trace(go.Scatter(x=x_values, y=simulation_df[col], mode='lines', name=f'Simulation {col+1}'))
        fig2.add_trace(go.Scatter(x=x_values, y=[data['Adj Close'][-1]]*(n_days+1), mode='lines', name='Last Price', line=dict(color='red', dash='dash')))
        fig2.update_layout(title=f'Monte Carlo Simulation for {ticker} Stock', xaxis_title='Day', yaxis_title='Price')
        st.plotly_chart(fig2)


    if __name__ == "__main__":
        main()



with Predictions:
    # Function to fetch stock data
    def fetch_stock_data(ticker, start_date, end_date):
        data = yf.download(ticker, start=start_date, end=end_date)
        return data

    # Function to plot the actual time series of Adjusted Close prices
    def plot_actual_time_series(df):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Adj Close'], mode='lines', name='Adjusted Closing Price'))
        fig.update_layout(title='Actual Time Series of Adjusted Close Prices', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig)

    # Function to plot actual vs predicted values
    def plot_actual_vs_predicted(df, forecast):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Adj Close'], mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted'))
        fig.update_layout(title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig)

    # Function to plot the components of the model
    # Function to plot the components of the model using Plotly
    def plot_model_components(model, forecast):
        components = ['trend', 'yearly', 'weekly', 'daily']
        fig = go.Figure()
        for component in components:
            if component in forecast:
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast[component], mode='lines', name=component.capitalize()))
        fig.update_layout(title='Model Components', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig)


    # Main Streamlit app
    def main():
        st.title('Stock Price Prediction with Prophet')
        
        # Input text box for user input
        ticker = st.text_input('Enter Stock Ticker', 'AAPL')
        
        # Fetching data
        end_date = pd.to_datetime('today')
        start_date = end_date - pd.DateOffset(years=1)
        data = fetch_stock_data(ticker, start_date, end_date)
        
        # Plot actual time series
        plot_actual_time_series(data)
        
        # Prepare data for Prophet
        df_prophet = data.reset_index().rename(columns={'Date': 'ds', 'Adj Close': 'y'})
        
        # Initialize the Model
        model = Prophet(daily_seasonality=False, yearly_seasonality=True)
        
        # Fit the Model
        model.fit(df_prophet)
        
        # Set the duration of predictions
        period = 365
        
        # Make a DataFrame to hold predictions
        future = model.make_future_dataframe(periods=period)
        
        # Predict
        forecast = model.predict(future)
        
        # Plot actual vs predicted values
        plot_actual_vs_predicted(data, forecast)
        
        # Plotting the components of the model
        plot_model_components(model, forecast)

    if __name__ == "__main__":
        main()
#with GoldenCross:
