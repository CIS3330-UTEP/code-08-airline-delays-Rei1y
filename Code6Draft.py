import pandas as pd #Github
import statsmodels.api as sm #Github
import matplotlib.pyplot as plt #Github
import numpy as np
import seaborn as sns

#-----------------------------------------------------------------------------------------------------------------

#DESCRIPTIVE ANALYTICS: OLS REGRESSION RESULTS - Referenced by CASA 15 and Lecture 16, Slide 7
filename = 'Flight_Delays_2018.csv' #Github


df = pd.read_csv(filename)

delay_df = pd.DataFrame({'ARR_DELAY': df['ARR_DELAY'], 'DEP_DELAY': df['DEP_DELAY']}) # Referenced from CASA 15 
# print(df)


y = delay_df['ARR_DELAY'] # Referenced from CASA 15 
x = delay_df['DEP_DELAY'] # Referenced from CASA 15 

x = sm.add_constant(x) # Referenced from CASA 15 

model = sm.OLS(y, x).fit() # Referenced from CASA 15 

print(model.summary()) # Referenced from CASA 15 

print("\nObtaining Parameters of the Linear Fit")
print(model.params)

#--------------------------------

#OLS REGRESSION VISUAL
a,b = np.polyfit(delay_df['DEP_DELAY'], delay_df['ARR_DELAY'], 1) # Referenced from CASA 15 

plt.scatter(delay_df['DEP_DELAY'], delay_df['ARR_DELAY'], color= 'purple') # Referenced from CASA 15 

plt.plot(delay_df['DEP_DELAY'], a*delay_df['ARR_DELAY']+b) # Referenced from CASA 15 

plt.text(1,90, 'y= ' + '{:.3f}'.format(b) + ' + {:.3f}'.format(a) + 'x', size=12) # Referenced from CASA 15 

plt.xlabel('DEP_DELAY') # Referenced from CASA 15 
plt.ylabel('ARR_DELAY') # Referenced from CASA 15 

plt.show() # Referenced from CASA 15 

#--------------------------------
# USEFUL PYTHON CODE - README
fig, ax = plt.subplots()
fig = sm.graphics.plot_fit(model, 1, ax=ax) # Referenced from AI
plt.show()

#-----------------------------------------------------------------------------------------------------------------

#BOXPLOTS
filter_query = "DEP_DELAY > 30" # Guided by AI but referenced by Lecture 15, Slide 6
filtered_df = df.query(filter_query) # Referenced by Lecture 15, Slide 6
# filtered_df = df_clean.query(filter_query) # Referenced by Lecture 15, Slide 6
filtered_df.boxplot(column='ARR_DELAY') # Referenced by Lecture 15, slide 6

plt.show()
#-----------------------------------------------------------------------------------------------------------------

#CORRELATION MATRIX
corr_matrix = df.corr(numeric_only=True).round(2) # Referenced by Lecture 17, Slide 6

sns.heatmap(corr_matrix, annot=True, vmax=1, vmin=-1, cmap ='vlag') # Referenced by Lecture 17, Slide 6: coolwarm (original color)

plt.show()

