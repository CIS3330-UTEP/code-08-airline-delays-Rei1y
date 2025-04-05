import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

filename = 'Flight_Delays_2018.csv'
df = pd.read_csv(filename)

# 1. Descriptive Analysis
# Example: Correlation analysis
numeric_cols = df.select_dtypes(include=['number']).columns
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()

# Example: Top 10 airports by flight count
top_airports = df['ORIGIN'].value_counts().nlargest(10).index
df_filtered = df[df['ORIGIN'].isin(top_airports)]

# 2. Predictive Analytics
# Example: Model with multiple predictors
predictors = ['DEP_DELAY', 'CARRIER_DELAY', 'DISTANCE', 'WEATHER_DELAY'] #add or remove predictors as needed.
df_filtered = df_filtered.dropna(subset=['ARR_DELAY'] + predictors) #Removes Nan values from needed columns.
X = df_filtered[predictors]
X = sm.add_constant(X)
y = df_filtered['ARR_DELAY']
model = sm.OLS(y, X).fit()
print(model.summary())

# Example: visualization
plt.scatter(df_filtered['DEP_DELAY'], df_filtered['ARR_DELAY'])
plt.plot(df_filtered['DEP_DELAY'], model.params[0] + model.params[1] * df_filtered['DEP_DELAY'], color='red')
plt.show()

# Add more visualizations and analysis as needed