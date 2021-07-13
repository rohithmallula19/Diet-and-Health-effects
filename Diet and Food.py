#In this project I examined how conventional wisdom stands up to data: does our diet determine our health outcomes? To characterize world diets, I use data on consumption of over 100 food items on a per country basis, from the United Nations Food and Agriculture Organization (FAO). For health outcomes, I use non-communicable disease data from the World Health Organization (WHO). After exploring and cleaning the data, I use statistical methods to examine whether world diets can explain variation in mortality rates across countries, and which food items are linked to disease.



#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import itertools
%matplotlib inline
#Load cardiovascular disease data
cardio_file = '../data/data.csv'
cardio_data = pd.read_csv(cardio_file)

#Select relevant field
cardio_data_all = cardio_data.drop(['Age-standardized mortality rate by cause (per 100 000 population).1', \
                  'Age-standardized mortality rate by cause (per 100 000 population).2'], axis=1)

cardio_years = ['2000', '2012']
cardio_data = []
#Create list of dataframes in 'cardio_data' according to 'cardio_years'
for i in np.arange(len(cardio_years)):
    #Isolate data from year of interest
    cardio_data.append(cardio_data_all[cardio_data_all['Unnamed: 1'].isin([cardio_years[i]])])
    #Drop year column
    cardio_data[i] = cardio_data[i].drop(['Unnamed: 1'], axis=1)
    #Rename country column
    cardio_data[i] = cardio_data[i].rename(columns={'Unnamed: 0' : 'Country'})
    
# Set disease dataframe indices as country to prepare for merging
cardio_data[0] = cardio_data[0].set_index(['Country'])
cardio_data[1] = cardio_data[1].set_index(['Country'])

#Show 2012 data
cardio_data[1].head()
#Mapping function 'map_quantity'
#Inputs
#country_vector:  List of country names (strings)
#quantity_vector:  Corresponding numerical array of quantities
#title:  Title for colorbar
def map_quantity(country_vector, quantity_vector, title):
    #Lists of uncommon country names and their replacements
    map_names_need_changed = [
    'Bolivia',
    "C�te d'Ivoire",
    'Republic of Congo',
    'United Kingdom',
    'The Gambia',
    'Iran',
    'Lao PDR',
    'Macedonia',
    'Dem. Rep. Korea',
    'Syria',
    'Tanzania',
    'United States',
    'Venezuela',
    'Vietnam']
    countries_to_replace = [
    'Bolivia (Plurinational State of)',
    "Cote d'Ivoire",
    'Democratic Republic of the Congo',
    'United Kingdom of Great Britain and Northern Ireland',
    'Gambia',
    'Iran (Islamic Republic of)',
    "Lao People's Democratic Republic",
    'The former Yugoslav republic of Macedonia',
    "Democratic People's Republic of Korea",
    'Syrian Arab Republic',
    'United Republic of Tanzania',
    'United States of America',
    'Venezuela (Bolivarian Republic of)',
    'Viet Nam']

    #Basic stats on data
    min_quant = min(quantity_vector)
    max_quant = max(quantity_vector)
    #Set up map elements and axes
    shapename = 'admin_0_countries'
    countries_shp = shpreader.natural_earth(resolution='110m',
                                            category='cultural', name=shapename)
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_axes([0.05, 0.15, 0.9, 0.8], projection=ccrs.Robinson())
    ax2 = fig.add_axes([0.05, 0.05, 0.9, 0.07])
    #Loop through countries in map database
    for country in shpreader.Reader(countries_shp).records():
        this_country_name = country.attributes['name_long']
        #Change name if necessary
        if this_country_name in map_names_need_changed:
                this_country_name = countries_to_replace[map_names_need_changed.index(this_country_name)]
        #If map country is in list of quantities, assign color based on quantity and map it
        if this_country_name in country_vector:
            #Get quantity for this country
            this_quantity = quantity_vector[country_vector.index(this_country_name)]
            #Scale quantity on [0, 1]
            scaled_quant = (this_quantity - min_quant)/(max_quant - min_quant)
            #Get scaled color for map
            this_color = plt.cm.jet(int(round(scaled_quant*256.)))
            #Plot country with this color
            ax1.add_geometries(country.geometry, ccrs.PlateCarree(),
                              facecolor=this_color)
        else:
            #Plot country in gray
            ax1.add_geometries(country.geometry, ccrs.PlateCarree(),
                              facecolor=np.array([(0.5, 0.5, 0.5)]))
            #And print the country to know missing countries
            #print this_country_name

    #Create plot
    cmap = mpl.cm.jet
    norm = mpl.colors.Normalize(vmin=min_quant, vmax=max_quant)
    cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')
    font_size = 18
    cb.set_label(title,size=18)
    cb.ax.tick_params(labelsize=font_size)
    plt.show()
    return None
#Select 0 for 2000 results, 1 for 2012
map_index = 0

country_names = cardio_data[map_index].index.values.tolist()

quantity_to_map = cardio_data[map_index]\
['Age-standardized mortality rate by cause (per 100 000 population)'].values.astype('float')

map_title = 'Age-standardized mortality rate of cardiovascular disease (per 100,000 population), '\
+ cardio_years[map_index]
    
map_quantity(country_names, quantity_to_map, map_title)



#CropData
crop_file = '../data/1caaffac-d01c-4d02-8735-404d3089b47e.csv'
crop_data = pd.read_csv(crop_file)
#Here I get a list of the unique crop food items and separate data into training and testing sets.
#Notice that some crop items are aggregates of others. For example, 'Fruits - Excluding Wine' is made up of other food items present in the list. In terms of the analysis methods used below, this would constitute multicollinearity of predictors, violating the assumptions of linear regression. Variable selection methods such as lasso would likely remove such correlated predictors, however I remove these manually as this is tractable here.

#Get food item names
#ItemNames
crop_items = crop_data.ItemName.unique()
#Get rid "nan" (last column), but save country populations (first column) for later.  The rest are food items
crop_items = crop_items[0:-1]
#Get rid of Grand Total
crop_items = crop_items[np.not_equal(crop_items, 'Grand Total')]
print crop_items

#Get "food supply" for the requested range of years.  The FAO calculates food supply as production minus exports
#plus imports, so this is a stand-in for consumption.
combined_mask = []
crop_subset = []
crop_year_range = [[1990, 2000], [2002, 2012]]
for i in np.arange(len(crop_year_range)):
    combined_mask.append( crop_data.ItemName.isin(crop_items) & \
        crop_data.ElementName.isin(['Food supply quantity (kg/capita/yr)', 'Total Population - Both sexes']) & \
        (crop_data.Year <= crop_year_range[i][1]) & \
        (crop_data.Year >= crop_year_range[i][0]) )
    crop_subset.append(crop_data[combined_mask[i]])

# crop_subset[0].head()

#Meat and Fish Data
#Load data
meat_fish_file = '../data/c1da0671-f39b-4d39-a421-6a19af9d6cd1.csv'
meat_fish_data = pd.read_csv(meat_fish_file)

#Explore data
meat_fish_data.head()

#ItemNames
meat_items = meat_fish_data.ItemName.unique()
#Get rid of "nan"
meat_items = meat_items[1:-1]
#Get rid of Grand Total
meat_items = meat_items[np.not_equal(meat_items, 'Grand Total')]
print meat_items
#Create subsets
#Get "food supply" for the requested range of years.  The FAO calculates food supply as production minus exports
#plus imports, so this is a stand-in for consumption.
combined_mask = []
meat_subset = []
for i in np.arange(len(crop_year_range)):
    combined_mask.append( meat_fish_data.ItemName.isin(meat_items) & \
        meat_fish_data.ElementName.isin(['Food supply quantity (kg/capita/yr)']) & \
        (meat_fish_data.Year <= crop_year_range[i][1]) & \
        (meat_fish_data.Year >= crop_year_range[i][0]) )
    meat_subset.append(meat_fish_data[combined_mask[i]])

#Make a list of countries
# meat_countries = meat_subset[0].AreaName.unique()
# print meat_countries

#Combining Food and Disease data
#Initialize list of lists, for both food types (crops and meat), each subset into training and testing data
#[value]*number creates a list of size number and initializes each member with value, in Python
#This will be a list of lists of dataframes:
food_means = [[None]*2 for i in range(2)]
# print food_means

#Calculate food means, dropping NaNs
for i in range(2): #training, then testing data
    food_means[0][i] = crop_subset[i].groupby(['AreaName', 'ItemName'], as_index=False).mean()
    food_means[1][i] = meat_subset[i].groupby(['AreaName', 'ItemName'], as_index=False).mean()

food_means[0][0].head()

#Initialize list of lists, for both food types (crops and meat) and training and testing data
food_means = [[None]*2 for i in range(2)]
# print food_means

#Calculate food means, dropping NaNs

for i in range(2): #training, then testing data
    food_means[0][i] = crop_subset[i].groupby(['AreaName', 'ItemName'], as_index=False).mean()
    food_means[1][i] = meat_subset[i].groupby(['AreaName', 'ItemName'], as_index=False).mean()
    
#Change country name so it's the same between food and disease data
for i in range(2):
    for j in range(2):
        #Change country names
        mask = food_means[i][j].isin(['United Kingdom'])
        food_means[i][j] = food_means[i][j].where(~mask, other='United Kingdom of Great Britain and Northern Ireland')
        
#Perform additional data cleaning and formatting:  drop unnecessary columns, set country as index
for i in range(2):
    for j in range(2):
        food_means[i][j] = food_means[i][j].drop(['AreaCode', 'ElementCode', 'ItemCode', 'Year'], axis=1)
        food_means[i][j] = food_means[i][j].pivot(index='AreaName', columns='ItemName', values='Value')

#Examine columns of dataframe
# food_means[0][0].columns
#I use the Pandas merge function first to combine the meat and crop data into one dataframe for all food, and then to combine the food and disease data. The final merge is an "inner" merge using the indices of country names from each data frame. The resulting data frame has the food and disease data for the countries in common between each data set.

#I also take this opportunity to create a dataframe that has no aggregate food items, to avoid multicollinearity of predictors.

#Merge meat and crop into one food data set
all_food = [None]*2
all_food_no_agg = [None]*2
for i in range(2):  #Training and testing data
    all_food[i] = pd.merge(food_means[0][i], food_means[1][i], right_index=True, left_index=True)
    all_food_no_agg[i] = all_food[i].drop(['Meat', 'Fish, Seafood',
                                   'Cereals - Excluding Beer', 'Oilcrops', 'Roots & Tuber Dry Equiv',
                                    'Starchy Roots', 'Sugar & Sweeteners', 'Vegetables', 'Pulses',
                                    'Nuts and products', 'Groundnuts (in Shell Eq)', 'Fruits - Excluding Wine',
                                    'Stimulants', 'Spices', 'Vegetable Oils', 'Alcoholic Beverages',
                                    'Sugar (Raw Equivalent)', 'Sugar, Raw Equivalent',
                                    'Rice (Paddy Equivalent)'], axis=1)

#There are a few more countries in 2000 than 2012, same number of food items
print all_food[0].shape, all_food[1].shape
all_food[1].head()
# all_food[1]['Population']

# Merge into final data frame
# Currently this throws out countries where the string name of the country doesn't match exactly
# Need to determine where this could be improved
#Change NaN to zero, representing assumption that unreported food items are of negligible consumption
#Put in table for regression

#Prior to merging, these can be used to check which countries are outside the intersection
#cardio_data[0].index[~cardio_data[0].index.isin(all_food[0].index)]
#all_food[0].index[~all_food[0].index.isin(cardio_data[0].index)]

final_df = [None]*2
final_df_no_agg = [None]*2
data_table = [None]*2
data_table_no_agg = [None]*2
pop = [None]*2
for i in range(2): #For training and testing dat
    final_df[i] = pd.merge(all_food[i], cardio_data[i], right_index=True, left_index=True)
    final_df[i].fillna(value=0, inplace=True)
    
    final_df_no_agg[i] = pd.merge(all_food_no_agg[i], cardio_data[i], right_index=True, left_index=True)
    final_df_no_agg[i].fillna(value=0, inplace=True)
    
#     #Put population data in array, then delete it from dataframe
    pop[i] = final_df[i]['Population'].values
    final_df[i] = final_df[i].drop(['Population'], axis=1)
    final_df_no_agg[i] = final_df_no_agg[i].drop(['Population'], axis=1)
    #Put rest of data frame (food items and disease rate) in array
    data_table[i] = final_df[i].values
    data_table_no_agg[i] = final_df_no_agg[i].values

#Put in X and Y arrays for regression.  Use data frames that contain no aggregate values.
X_train = data_table_no_agg[0][:,0:-1]
Y_train = data_table_no_agg[0][:,-1]
pop_train = pop[0]
X_test = data_table_no_agg[1][:,0:-1]
Y_test = data_table_no_agg[1][:,-1]
pop_test = pop[1]

#Check item count
# result_2000.count()
#Examine data frame
final_df_no_agg[0].head()
#Put all item names from training data in an array for examining predictors
all_food_items_array = np.array(final_df_no_agg[0].columns.values)
all_food_items_array = np.delete(all_food_items_array, -1) #remove response variable
# print all_food_items_array
print np.shape(all_food_items_array)
print np.shape(X_train)

#Get country names for later use
final_df_countries = final_df_no_agg[0].index.values

# print final_df_countries
# print final_df_countries, pop_train

Analysis 
#I try multiple linear regression as a simple initial approach, as well as lasso regression. I use the scikit-learn package for all analyses.


from sklearn import linear_model
import sklearn
clf = linear_model.LinearRegression()
clf.fit(X_train, Y_train)
print 'Linear regression'
print 'Train R^2 = ', clf.score(X_train, Y_train), ', Test R^2 = ', clf.score(X_test, Y_test)

#A linear regression using all 104 food items has a high training score, explaining 85% of the variance in the data. However the testing score is substantially lower at $R^2 = 0.11$, indicating the model describes the data scarcely better than the average of all the responses would. This indicates that the linear regression is overfit to the training data, and is not useful to predict the new, unseen observations in the testing data. In statistical terms, it has high variance, and low bias.

#Next I examine a lasso regression, which allows increased bias in the hopes that a lower variance will lead to less overfitting of the training data. Lasso performs variable selection, which will potentially cut down the large list of predictors to a more managable and interpretable set.

#Lasso regression works by including the sum of the absolute values of regression coefficients along with the sum of squared errors in the cost function, which is minimized to perform the regression fit. The sum of coefficient absolute values is scaled by a parameter, alpha, in the cost function.

#I try several different values of alpha to find which gives the best model testing score, displaying the results as text output and a plot of training score versus alpha:

alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
clf2 = [None]*len(alphas)
train_score = [None]*len(alphas)
test_score = [None]*len(alphas)

for i in range(len(alphas)):
    clf2[i] = sklearn.linear_model.Lasso(max_iter=10000, alpha=alphas[i])
    clf2[i].fit(X_train, Y_train)
    train_score[i] = clf2[i].score(X_train, Y_train)
    test_score[i] = clf2[i].score(X_test, Y_test)
    print 'Lasso regression, alpha = ' + str(alphas[i])
    print 'Train R^2 = ', clf2[i].score(X_train, Y_train), ', Test R^2 = ', clf2[i].score(X_test, Y_test)

#Plot testing score versus alphas
plt.semilogx(alphas, test_score)
plt.xlabel('alpha')
plt.ylabel('testing R^2')

#Testing score is highest at alpha = 10. Here the lasso regression has a slightly lower training score than ordinary least squares regression. However the testing score is higher, indicating lasso can explain about half the variance in the new, unseen data of the testing set. This seems like fairly successful performance, given the other factors such as exercise habits that may also affect rates of cardiovascular disease.
#I examine which coefficients were retained by the lasso, and their associations with cardiovascular disease, by making a bar chart of the non-zero coefficients:


#Get coefficient values for regression with highest testing score
optimal_index = np.argmax(test_score)
coefs2 = clf2[optimal_index].coef_
#Table of coefficients
# zip(all_food_items_array, coefs2)

#Make bar chart of non zero coefficients
non_zero_coeffs2 = coefs2[np.nonzero(coefs2)]
non_zero_items = all_food_items_array[np.nonzero(coefs2)]
# print np.shape(non_zero_coeffs2)
# print np.shape(non_zero_items)
fig = plt.figure(figsize=(20,10))
ax = plt.axes()

ind = np.arange(len(non_zero_coeffs2))  # the x locations for the groups
width = 0.5       # the width of the bars
ax.bar(ind, non_zero_coeffs2, width)
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(non_zero_items)
labels = ax.get_xticklabels()
font_size = 14
plt.setp(labels, rotation=90, fontsize=font_size)
plt.ylabel('Non-zero lasso coefficients', fontsize=font_size)
ax.grid(True)
plt.show

#Food items with the largest positive coefficients have positive associations with mortality from cardiovascular disease. These include "Beverages, Alcoholic", a category that apparently includes distilled alcohol, but not wine or beer, as well as lamb and goat meats, classified as red meats. The largest negative association by far is olive oil, a known health food. Crustaceans, such as shrimp, crab, and lobsters, are the next most negative, although non-centrifugal sugar, described as a "traditional product" as opposed to the "industrialized alternative" of refined sugar, is similar
#Intuitively, it seems like global consumption patterns in these foods should match up with disease rates. To check this, I examine the two foods with the strongest negative associations in maps:

#Map olive oil consumption from training data
country_names = final_df_countries.tolist()
quantity_to_map = final_df_no_agg[0]['Olive Oil'].values
map_title = 'Olive oil consumption (kg/capita/yr)'
map_quantity(country_names, quantity_to_map, map_title)

#Map crustacean consumption from training data
country_names = final_df_countries.tolist()
quantity_to_map = final_df_no_agg[0]['Crustaceans'].values
map_title = 'Crustacean consumption (kg/capita/yr)'
map_quantity(country_names, quantity_to_map, map_title)

#Olive oil is most highly consumed in the Mediterranean region, while crustaceans are generally popular in island nations, and to some extent in the U.S., Australia and China. Revisiting the map of cardiovascular mortality rates, it is apparent that these regions do indeed have lower rates, providing confidence in the results:

#Map cardiovascular mortality again
#Select 0 for 2000 results, 1 for 2012
map_index = 0

country_names = cardio_data[map_index].index.values.tolist()

quantity_to_map = cardio_data[map_index]\
['Age-standardized mortality rate by cause (per 100 000 population)'].values.astype('float')

map_title = 'Age-standardized mortality rate of cardiovascular disease (per 100,000 population), '\
+ cardio_years[map_index]
    
map_quantity(country_names, quantity_to_map, map_title)

#Conclusion 
#Variable selection with the lasso provided improved results over ordinary least squares regression, in terms of the testing model fit. The lasso coefficients enabled interpretation and observation about what kinds of food are associated with cardiovascular disease mortality.
