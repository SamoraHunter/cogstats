import ast
import datetime as dt
from pstats import Stats

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fitter import Fitter, get_common_distributions, get_distributions
from pandas import json_normalize
from sklearn import linear_model
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

#import pandas_profiling

#sys.path.insert(0, '/home/jovyan/work/Samora/')


def parse_meta_anns_json_column(df, column_name='meta_anns'):
    """
    Parse the JSON strings in the specified column of the DataFrame and
    create separate columns for each variable in the JSON entries.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column_name (str): Name of the column containing JSON strings. Default is 'meta_anns'.

    Returns:
    - pd.DataFrame: DataFrame with parsed JSON columns appended.
    """
    # Convert the string representation of a dictionary to a dictionary
    df[column_name] = df[column_name].apply(ast.literal_eval)

    # Normalize the JSON column
    df_normalized = json_normalize(df[column_name])

    # Concatenate the original DataFrame with the normalized JSON columns
    df = pd.concat([df, df_normalized], axis=1)

    # Drop the original JSON column
    df = df.drop(column_name, axis=1)

    return df


def batchLinearRegression(dataFrame, dependentVariableList, independentVariableList):
    
    data = dataFrame[independentVariableList]
    
    for dependentVariable in dependentVariableList:
        X_train, X_test, y_train, y_test = train_test_split(dataFrame, data,random_state = 0,test_size=0.25)
        regr = linear_model.LinearRegression()
        regr.fit(X_train,y_train)
        y_pred = regr.predict(X_train)

        print("R squared: {}".format(r2_score(y_true=y_train,y_pred=y_pred)))



def printMissingValues(dataFrame):
    for col in dataFrame.columns:
    
        print(dataFrame[col].name, dataFrame[col].isnull().sum() * 100 / len(dataFrame[col]))






def generateProfileReport(dataFrame, strName):
    """Pass data frame object, and name (string) for the html report. Can be found in current path directory
    as 'Pandas Profiling Report — '+strName+'.html' """
    
    profile = pandas_profiling.ProfileReport(dataFrame)
    profile.to_file(output_file='Pandas Profiling Report — '+strName+'.html')
    return profile


def getVariableName(variable, globalVariables):
    """Pass variable and globals.copy(), for example: getVariableName(elem,globals().copy())"""
    results = []
   
    for globalVariable in globalVariables:
        if id(variable) == id(globalVariables[globalVariable]):
            results.append(globalVariable)
            
    for result in results:
        if result == 'elem':
            results.remove(result)
            
    return str(results[0])


def plotDensityOverlay(listOfDataframes, colName, globalsCopy):
    """plotDensityOverlay(listOfDataframes, colName, globals.copy()) for example: plotDensityOverlay(allSmoke, 'SR_CURRENT_DOSE', globals().copy())"""
    for elem in listOfDataframes:
    
        ax = elem[colName].value_counts().plot(kind='density', legend=True)

        ax.set_title(colName +" on: "+ "  ".join([getVariableName(x,globalsCopy) for x in listOfDataframes]) )
        ax.legend([getVariableName(x,globalsCopy) for x in listOfDataframes])


def plotHistOverlay(listOfDataframes, colName, globalsCopy):
    """plotDensityOverlay(listOfDataframes, colName, globals.copy()) for example: plotDensityOverlay(allSmoke, 'SR_CURRENT_DOSE', globals().copy())"""
    for elem in listOfDataframes:
    
        ax = elem[colName].value_counts().plot(kind='hist', legend=True)

        ax.set_title(colName +" on: "+ "  ".join([getVariableName(x,globalsCopy) for x in listOfDataframes]) )
        ax.legend([getVariableName(x,globalsCopy) for x in listOfDataframes])


def fitCommonDist(dataColumn, timeout=60, additional = []):
    if(len(additional)>0):
        distributionsToCheck = get_common_distributions() + additional
    else:
        distributionsToCheck = get_common_distributions()
        
    f = Fitter(dataColumn, distributions = distributionsToCheck
    ,timeout = timeout)
    f.fit()
    # may take some time since by default, all distributions are tried
    # but you call manually provide a smaller set of distributions
    print(f.get_best())
    display(f.summary())


def fitAllDist(dataColumn, timeout=60):
    f = Fitter(dataColumn, distributions = get_distributions()
    ,timeout = timeout)
    f.fit()
    # may take some time since by default, all distributions are tried
    # but you call manually provide a smaller set of distributions
    print(f.get_best())
    display(f.summary())


def standardize(observation):
    z = (observation-np.mean(observation))/np.std(observation)
    return z

def get_quantiles(data):
    quantiles = []
    for q in np.arange(0, 1.001, 0.001):
        quantiles.append(np.quantile(data, q))
    return quantiles    

def pyqqplot(data1, data2):
    plt.figure()
    data1 = standardize(data1)
    data2 = standardize(data2)
    q1 = np.array(get_quantiles(data1))
    q2 = np.array(get_quantiles(data2))
    plt.scatter(q1, q2 )

    minim = min(data1.min(), data2.min())
    maxim = max(data1.max(), data2.max())
    plt.plot([minim, maxim], [minim, maxim], 'r-')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


def qqNormalPlot(x1):
    #x1 = np.random.normal(0, 1, size = 10000 )
    x2 = np.random.normal(0, 1, size=10000 )
    plt.figure()
    sns.distplot(x1, hist=False, kde=True)
    sns.distplot(x2, hist=True, kde= True, hist_kws={'edgecolor': 'black'})
    plt.legend(['x1','x2'])
    pyqqplot(x1, x2)
    plt.figure()
    plt.boxplot([x1,x2])


def qqLaplacePlot(x1):
    x2 = np.random.laplace(0, 5, size = 10000 )
    x1 = standardize(x1)
    x2 = standardize(x2)
    plt.figure()
    sns.distplot(x1, hist=False, kde=True)
    sns.distplot(x2, hist=True, kde= True, hist_kws={'edgecolor': 'black'})
    plt.legend(['x1','x2'])
    pyqqplot(x1,x2)
    plt.figure()
    plt.boxplot([x1,x2])



def qqGammaPlot(x1):
    x2 = np.random.gamma(2, 1, size=10000 )
    x1 = standardize(x1)
    x2 = standardize(x2)
    plt.figure()
    sns.distplot(x1, hist=False, kde=True)
    sns.distplot(x2, hist=True, kde= True, hist_kws={'edgecolor': 'black'})
    plt.legend(['x1','x2'])
    pyqqplot(x1,x2)
    plt.figure()
    plt.boxplot([x1,x2])

def batchPlotCooksDistance(df, model):
    """batchPlotCooksDistance(x, model)"""
    for colName in df.columns:
        #create instance of influence
        influence = model.get_influence()

        #obtain Cook's distance for each observation
        cooks = influence.cooks_distance

        #display Cook's distances
        #print(cooks)
        plt.scatter(df[colName], cooks[0])
        plt.xlabel(colName)
        plt.ylabel('Cooks Distance')
        plt.show()


def imputeMultipleDataFrame(dataFrame):
    imputer = IterativeImputer( n_nearest_features=None, imputation_order='ascending', missing_values = np.nan) #If None, all features will be used.
    imp = IterativeImputer(max_iter=10, random_state=0)
    imp.fit(dataFrame)
    trans = imp.transform(dataFrame)
    
    to_return = pd.DataFrame(trans, columns=dataFrame.columns)  
    return to_return


# def batchLinearRegression(independentDataFrame, dependentDataFrame, dependentVariableList, independentVariableList):

#     data = independentDataFrame[independentVariableList]
#     print(len(independentDataFrame), len(dependentDataFrame))
#     breakpoint()
#     for dependentVariable in dependentDataFrame.columns:

#         X_train, X_test, y_train, y_test = train_test_split((independentDataFrame), (dependentDataFrame[dependentVariable]),random_state = 0,test_size=0.25)

#         regr = linear_model.LinearRegression()
#         regr.fit(X_train,y_train)
#         y_pred = regr.predict(X_train)

#         print("R squared: {}".format(r2_score(y_true=y_train,y_pred=y_pred)))

def imputeKNNDataFrame(dataFrame):
    dataframe = variablesForImputation


    # define imputer
    imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')


    # fit on the dataset
    imputer.fit(dataframe)

    # transform the dataset
    Xtrans = imputer.transform(dataframe)

    return pd.DataFrame(trans, columns = dataframe.columns)


def batchQQCheck(dataFrame, colList):
    for colName in colList:
        if(len(dataFrame[colName].value_counts())>2):
            try:
                QQPlot(dataFrame[colName])
            except Exception as e:
                print(e)
                
            shapiro_test = Stats.shapiro(dataFrame[colName])
            print(shapiro_test)

def batchPairPlot(dataFrame, outcomeVariableList, colList):

    for outcomeVar in outcomeVariableList:
        
        g= sns.pairplot(dataFrame, x_vars=colList, y_vars=outcomeVar,  aspect=.7)
        g.fig.set_size_inches(40, 2)

def QQPlot(data):
    # seed the random number generator
    #seed(1)
    # generate univariate observations
    #data = 5 * randn(100) + 

    print(data.name)
    # q-q plot
    qqplot(data, line='r')
    pyplot.show()

def scatterOverlayTwo(dataFrame1, dataFrame2, label1, label2, col1, col2 ):
    
    ax1 = dataFrame1.plot(kind = 'scatter' , x =col1, y = col2, color = 'green', label = label1)
    ax2 = dataFrame2.plot(kind = 'scatter' , x =col1, y = col2, color = 'red', ax = ax1, label = label2)

def batchMannWhitneyU(group1, group2):
    for colName in group1.columns:
        if(group1[colName].value_counts().sum() >4 and group1[colName].nunique() >3 and group1[colName].dtypes !=object):
            try:
                print("---------------------------------------------------------------------------------")
                printmd(colName)
                print(f"{colName}::, {group1[colName].dtypes}, {group2[colName].dtypes}")

                print(ks_2samp(group1[colName], group2[colName]))

                results = mwu(group1[colName], group2[colName], 
                       alternative='two-sided')

                display(results)

                # given
                conf = 0.95
                colName = colName
                sample1 = group1[colName].dropna()
                sample2 = group2[colName].dropna()

                # mann-whitney
                
                stat, p = mannwhitneyu(sample1, sample2, use_continuity=False, alternative='two-sided')
                print(p)
                print(f"p= {round(p, 3)}")

                if p<=(1-conf):
                    print(f"Reject H0. {colName} looks likely to be different in each group.")
                else:
                    print(f"Failed to reject H0. {colName} does not look likely to be different in each group.")

                # confidence interval
                diff_median = np.median(list(map(operator.sub, sample1, sample2)) )
                lower, upper = non_param_unpaired_CI(sample1, sample2, conf)

                print(diff_median)
                print(f"{100*conf} % confidence that the median of the samples' diff lies b/w {lower} and {upper}.")


            except Exception as e:
                print("Passed")
                print(f"Failed on {colName} with : {e}")


def df_column_uniquify(df):
    df_columns = df.columns
    new_columns = []
    for item in df_columns:
        counter = 0
        newitem = item
        while newitem in new_columns:
            counter += 1
            newitem = "{}_{}".format(item, counter)
        new_columns.append(newitem)
    df.columns = new_columns
    return df


def genMissingValueDF(df):
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_name': df.columns,
                                     'percent_missing': percent_missing})

    return missing_value_df


def genMissingValueDFCustom(df, colNames):
    
    df = df[colNames]
    
    percent_missing = df.isnull().sum() * 100 / len(df)
    
    
    
    missing_value_df = pd.DataFrame({'column_name': colNames,
                                     'percent_missing': percent_missing})

    return missing_value_df

def ratioFunction(num1, num2):
    #num1 = input('Enter the first number: ')
    if(num1 == 0 or num2 ==0):
        #print("zero found")
        return np.nan
    
    num1 = float(num1) # Now we are good
    #num2 = input('Enter the second number: ')
    num2 = float(num2) # Good, good
    ratio12 = float(num1/num2)
    #print('The ratio of', str(num1), 'and', str(num2),'is', ratio12 + '.')
    return ratio12

def getCommonTwoLists(list1, list2):
    return list(set(list1).intersection(list2))


def getPStatFormat(p, stat):
    print("p=",p, " X2=", stat)
    if p < 0.01:
        print("χ2 =", round(stat), ";", "p < 0.01")
    elif p < 0.05:
        print("χ2 =", round(stat), ";", "p < 0.05")
    elif p > 0.05:
        print("χ2 =", round(stat), ";", "p > 0.05")


def df_column_uniquify(df):
    df_columns = df.columns
    new_columns = []
    for item in df_columns:
        counter = 0
        newitem = item
        while newitem in new_columns:
            counter += 1
            newitem = "{}_{}".format(item, counter)
        new_columns.append(newitem)
    df.columns = new_columns
    return df

def getNfeaturesANOVAF(X, y, n):
    
    res = []
    for colName in X.columns:
    
        res.append((colName, sklearn.feature_selection.f_classif(np.array(X[colName]).reshape(-1,1), y)[0]))
        
    sortedList = sorted(res, key=lambda x: x[1])
    
    sortedList.reverse()
    
    nFeatures = sortedList[:n]
    
    finalColNames = []
    for elem in nFeatures:
        finalColNames.append(elem[0])

#     for elem in nFeatures:
#         print(elem[0], float(elem[1]))

    
    
    return finalColNames


def convertclientvisit_dischargedtmToDT(date_time_str):
    """Of the form '2020-04-16T13:40:00'"""
    if(type(date_time_str) == str):
        date_time_str = date_time_str.split(".")[0]

        date_time_obj = dt.datetime.strptime(date_time_str, '%Y-%m-%dT%H:%M:%S')
        return date_time_obj

def convertclientvisit_admitdtmToDT(date_time_str):
    """Of the form '2020-03-05 00:44:00+00:00'"""
    if(type(date_time_str) == str):
        date_time_str = date_time_str.split("+")[0]

        date_time_obj = dt.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
        return date_time_obj


def logit(p):
    logit_value = math.log(p / (1-p))
    return logit_value


def calcBMI(W, H):
    return W / ((H/100)**2)

def df_column_uniquify(df):
    df_columns = df.columns
    new_columns = []
    for item in df_columns:
        counter = 0
        newitem = item
        while newitem in new_columns:
            counter += 1
            newitem = "{}_{}".format(item, counter)
        new_columns.append(newitem)
    df.columns = new_columns
    return df

def model_to_equation_median(df, model):
    eqStr = []
    for i in range(0, len(df.columns)):
        
        eqStr.append("".join(['(',str(df[df.columns[i]].median()), '*', str(model.params[i]), ') + ']))
    allstr= "".join(eqStr)
        
    return allstr[:-2]

def model_to_equation_name(df, model):
    eqStr = []
    for i in range(0, len(df.columns)):
        
        eqStr.append("".join(['(',str(df.columns[i]), ' * ', str(round(model.params[i], 4)), ') + ']))
    allstr= "".join(eqStr)
        
    return allstr[:-2]


def model_to_equation_median_prob(df, model):
    eqStr = []
    for i in range(0, len(df.columns)):
        eqStr.append((df[df.columns[i]].median() * model.params[i]))

    #print(sum(eqStr))
    p = np.exp(sum(eqStr))/1+np.exp(sum(eqStr))
    #print(p)
        
    return p

def model_to_equation_median_prob_single(df, model):
    eqStr = []
    #for i in range(0, len(df.columns)):
    eqStr.append((df.median() * model.params[0]))

    #print(sum(eqStr))
    p = np.exp(sum(eqStr))/1+np.exp(sum(eqStr))
    #print(p)
        
    return p

"""Script takes a csv column of free text entries in an ethnicity field

It attempts to map these entries into UK census categories

Groups derived from https://www.ethnicity-facts-figures.service.gov.uk/style-guide/ethnic-groups

No fuzzy matching is used, add edge cases below in script if unusual behaviour. 

Some countries groups have been modified to catch more correct mappings. 

Countries are assumed as containing a single ethnic group for simplicity of the script, these are categorised as belonging to the default ethnic group unless user specifies, for example "Asian Caribbean". 

Categorisation lists and output lists contain errors and ambiguities, manually look over result outputs. 

Explicit user specification of a racial group, example: "White" takes precidence over national origin term. """



def get_odds_ratio_conf_intervals(model):
    params = log_reg_imputed.params
    conf = log_reg_imputed.conf_int()
    conf['Odds Ratio'] = params
    conf.columns = ['5%', '95%', 'Odds Ratio']
    print(np.exp(conf))
    return np.exp(conf)



class EthnicityAbstractor:
    def abstractEthnicity(dataFrame, outputNameString, ethnicityColumnString):
        
        """'abstractEthnicity(dataFrame, outputNameString, ethnicityColumnString)'"""
        
        assumeBritishWhite = True
        assumeEnglishWhite = True
        assumeEuropeanWhite = True
        assumeAfricanBlack = True
        assumeAsianAsian = True
        assumeSouthAmericanOther = True
        assumeNorthAmericanOther = True
        includeNationalitiesForCountries = True 
        edgeCases = True
        
        
        #targetList = pd.read_csv('allanemia.csv')
        
        targetList = dataFrame
        
        #fileOutputName = 'allanemiarace.csv'
        
        fileOutputName = outputNameString + '.csv'
        
        targetList.columns
        
        #targetColumnString = 'Eth'
        
        targetColumnString = ethnicityColumnString
        
        
        additionalColumsToAppend = ['client_idcode']
        
        additionalColumnsToAppend = targetList.columns
        
        len(targetList[targetColumnString].unique())
        
        
        len(targetList[targetColumnString])
        
        targetList[targetColumnString].nunique()
        
        racecodeEntries = targetList[targetColumnString].tolist()

        racecodeEntries = pd.DataFrame(racecodeEntries, columns =[targetColumnString])

        #print(len(racecodeEntries[targetColumnString]))
        #racecodeEntries[targetColumnString].dropna(axis=0,inplace=True)
        racecodeEntries[targetColumnString] = racecodeEntries[targetColumnString].fillna('other_ethnic_group')
        #print(len(racecodeEntries[targetColumnString]))
        
        #print(len(racecodeEntries))
        
        #racecodeEntries.dropna(inplace = True)
        
        #racecodeEntries.reset_index(inplace = True)
        
        #df_testMap = pd.DataFrame(racecodeEntries)
        df_testMap = dataFrame[['client_idcode', ethnicityColumnString]].copy()
        
        for col in dataFrame.columns:
            df_testMap[col] = dataFrame[col]
            
        df_testMap.insert(1, 'census', 'other_ethnic_group')
        
        
        
        df_testMap
        
        #Set default value as other
        df_testMap['census'][0]=('other_ethnic_group')
        
        whiteList = ['English', 
         'Welsh',
         'Scottish',
         'Northern',
         'Irish',
         'British',
         'Irish',
         'Gypsy',
         'Irish Traveller',
         'Any other White background']

        mixedOrMultipleEthnicGroups = ['White and Black Caribbean',
        'White and Black African', 
        'White and Asian',
        'Any other Mixed or Multiple ethnic background']

        asianOrAsianBritish = ['Indian', 
                               'Pakistani',
                               'Bangladeshi',
                               'Chinese', 
                               'Any other Asian background']

        blackOrAfricanOrCaribbeanOrBlackBritish = ['African',
        'Caribbean',
        'Any other Black, African or Caribbean background'
        ]

        Arab = ['Arab', 
               'Any other ethnic group']

        censusList = [

        whiteList, 
        mixedOrMultipleEthnicGroups, 
        asianOrAsianBritish, 
        blackOrAfricanOrCaribbeanOrBlackBritish,
        Arab  
        ]

        #Groups derived from https://www.ethnicity-facts-figures.service.gov.uk/style-guide/ethnic-groups

        blackList = ['black', 'african', 'caribbean', 'black british', 'black african', 'black carribean']

        whiteList = ['white', 'caucasian', 'gypsy', 'traveller', 'other white', 'white other']

        asianList = ['asian', 'chinese', 'pakistani', 'bangladeshi', 'indian']

        otherList = ['arab', 'not specified']

        mixedList = ['mixed', 'multiple', 'biracial', 'multiracial', 'white and asian', 'white and black', 'white and hispanic', 'black and white', 'asian and white', 'hispanic and white' ]


        allEthList = blackList + whiteList + asianList + otherList + mixedList
        
        africanCountries = ['algeria',
         'angola',
         'benin',
         'botswana',
         'burkina',
         'faso',
         'burundi',
         'cabo',
         'verde',
         'cameroon',
         'central african republic',
         'chad',
         'comoros',
         'congo,',
         'democratic',
         'republic of the congo',
         'republic of the cote d\'ivoire',
         'djibouti',
         'egypt',
         'equatorial',
         'guinea',
         'eritrea',
         'eswatini',
         'ethiopia',
         'gabon',
         'gambia',
         'ghana',
         'guinea',
         'guinea-bissau',
         'kenya',
         'lesotho',
         'liberia',
         'libya',
         'madagascar',
         'malawi',
         'mali',
         'mauritania',
         'mauritius',
         'morocco',
         'mozambique',
         'namibia',
         'niger',
         'nigeria',
         'rwanda',
         'sao tome and principe',
         'senegal',
         'seychelles',
         'sierra',
         'leone',
         'somalia',
         'south',
         'africa',
         'south',
         'sudan',
         'sudan',
         'tanzania',
         'togo',
         'tunisia',
         'uganda',
         'zambia',
         'zimbabwe']

        asianCountries = ['afghanistan',
         'armenia',
         'azerbaijan',
         'bahrain',
         'bangladesh',
         'bhutan',
         'brunei',
         'cambodia',
         'china',
         'cyprus',
         'east',
         'timor',
         'egypt',
         'georgia',
         'india',
         'indonesia',
         'iran',
         'iraq',
         'israel',
         'japan',
         'jordan',
         'kazakhstan',
         'kuwait',
         'kyrgyzstan',
         'laos',
         'lebanon',
         'malaysia',
         'maldives',
         'mongolia',
         'myanmar',
         'nepal',
         'north',
         'korea',
         'oman',
         'pakistan',
         'palestine',
         'philippines',
         'qatar',
         'russia',
         'saudi arabia',
         'singapore',
         'south korea',
         'sri lanka',
         'syria',
         'taiwan',
         'tajikistan',
         'thailand',
         'turkey',
         'turkmenistan',
         'united arab emirates',
         'uzbekistan',
         'vietnam',
         'yemen']

        europeanCountries =['albania',
         'andorra',
         'armenia',
         'austria',
         'azerbaijan',
         'belarus',
         'belgium',
         'bosnia and herzegovina',
         'bulgaria',
         'croatia',
         'cyprus',
         'czechia',
         'denmark',
         'estonia',
         'finland',
         'france',
         'georgia',
         'germany',
         'greece',
         'hungary',
         'iceland',
         'ireland',
         'italy',
         'kazakhstan',
         'kosovo',
         'latvia',
         'liechtenstein',
         'lithuania',
         'luxembourg',
         'malta',
         'moldova',
         'monaco',
         'montenegro',
         'netherlands',
         'north macedonia',
         'norway',
         'poland',
         'portugal',
         'romania',
         'russia',
         'san marino',
         'serbia',
         'slovakia',
         'slovenia',
         'spain',
         'sweden',
         'switzerland',
         'turkey',
         'ukraine',
         'united kingdom',
         'vatican city']

        northAmericanCountries = ['antigua and barbuda',
         'bahamas',
         'barbados',
         'belize',
         'canada',
         'costa rica',
         'cuba',
         'dominica',
         'dominican republic',
         'el salvador',
         'grenada',
         'guatemala',
         'haiti',
         'honduras',
         'jamaica',
         'mexico',
         'nicaragua',
         'panama',
         'saint kitts and nevis saint lucia',
         'saint vincent and the grenadines',
         'trinidad and tobago',
         ]

        southAmericanCountries = ['argentina',
         'bolivia',
         'brazil',
         'chile',
         'colombia',
         'ecuador',
         'guyana',
         'paraguay',
         'peru',
         'suriname',
         'uruguay',
         'venezuela']

        africanNationalities = ['Swazi',
         'algerian',
         'angolan',
         'beninese',
         'botswanan',
         'burkinese',
         'burundian',
         'cameroonian',
         'cape verdeans',
         'chadian',
         'congolese',
         'djiboutian',
         'egyptian',
         'eritrean',
         'ethiopian',
         'gabonese',
         'gambian',
         'ghanaian',
         'guinean',
         'kenyan',
         'krio people',
         'liberian',
         'libyan',
         'madagascan',
         'malagasy',
         'malawian',
         'malian',
         'mauritanian',
         'mauritian',
         'moroccan',
         'mozambican',
         'namibian',
         'nigerian',
         'nigerien',
         'rwandan',
         'senegalese',
         'somali',
         'sudanese',
         'tanzanian',
         'togolese',
         'tunisian',
         'ugandan',
         'zambian',
         'zimbabwean',
         'african'
                               ]

        asianNationalities = [
         'afghan',
         'afghanistan',
         'armenian',
         'azerbaijani',
         'bahrain',
         'bahraini',
         'bangladesh',
         'bangladeshi',
         'bhutan',
         'bhutanese',
         'brunei',
         'burma',
         'burmese',
         'cambodia',
         'cambodian',
         'chinese',
         'filipino',
         'indian',
         'indonesian',
         'iranian',
         'iraqi',
         'japanese',
         'jordanian',
         'kazakh',
         'kuwaiti',
         'laotian',
         'lebanese',
         'malawian',
         'malaysian',
         'maldivian',
         'mongolian',
         'myanmar',
         'nepalese',
         'omani',
         'pakistani',
         'philippine',
         'qatari',
         'russian',
         'singaporean',
         'sri lankan',
         'syrian',
         'tadjik',
         'taiwanese',
         'tajik',
         'thai',
         'turkish',
         'turkmen',
         'turkoman',
         'uzbek',
         'vietnamese',
         'yemeni',
         'punjabi',
         'kurdish',
         'tamil',
         'kashmiri',
         'sinhala',
         'sinhalese'


        ]

        europeanNationalities = [
         'albanian',
         'andorran',
         'armenian',
         'australian',
         'austrian',
         'azerbaijani',
         'belarusan',
         'belarusian',
         'belgian',
         'bosnian',
         'brit',
         'british',
         'bulgarian',
         'croat',
         'croatian',
         'cypriot',
         'czech',
         'danish',
         'dutch',
         'english',
         'estonian',
         'finnish',
         'french',
         'georgian',
         'german',
         'greek',
         'holland',
         'hungarian',
         'icelandic',
         'irish',
         'italian',
         'latvian',
         'lithuanian',
         'maltese',
         'moldovan',
         'monacan',
         'montenegrin',
         'monégasque',
         'netherlands',
         'norwegian',
         'polish',
         'portuguese',
         'romanian',
         'scot',
         'scottish',
         'serb',
         'serbian',
         'slovak',
         'slovene',
         'slovenian',
         'spanish',
         'swedish',
         'swiss',
         'ukrainian',
         'welsh',
         'yugoslav'
         'ussr',
         'soviet',
         'cornish'

        ]

        northAmericanNationalities = [
         'bahamian',
         'barbadian',
         'belizean',
         'costa rican',
         'cuban',
         'dominican',
         'grenadian',
         'guatemalan',
         'haitian',
         'honduran',
         'mexican',
         'nicaraguan',
         'panamanian',
         'paraguayan',
         'salvadorean',
         'trinidadian'

        ]

        southAmericanNationalities = [

         'argentinian',
         'armenian',
         'bolivian',
         'brazilian',
         'chilean',
         'colombian',
         'ecuadorean',
         'ghanaian',
         'guyanese',
         'nicaraguan',
         'paraguayan',
         'peruvian',
         'surinamese',
         'uruguayan',
         'venezuelan'
        ]




        if(assumeBritishWhite):
            whiteList.append('british')


        if(assumeEnglishWhite):
            whiteList.append('english')



        if(assumeEuropeanWhite):
            whiteList = whiteList + europeanCountries


        if(assumeAfricanBlack):
            blackList = blackList + africanCountries


        if(assumeAsianAsian):
            asianList = asianList + asianCountries


        if(assumeSouthAmericanOther):
            otherList = otherList + southAmericanCountries



        if(assumeNorthAmericanOther):
            otherList = otherList + northAmericanCountries    




        if(includeNationalitiesForCountries):
            whiteList = whiteList + europeanNationalities


            blackList = blackList + africanNationalities

            asianList = asianList + asianNationalities

            otherList = otherList + southAmericanCountries + northAmericanCountries



        if(edgeCases):

            extraWhite = ['australian', 'american', 'usa', 
                             'united states', 'the united states of america', 
                             'canadian']
            whiteList = whiteList + extraWhite


        allEthList = blackList + whiteList + asianList + otherList + mixedList    

        #print(len(racecodeEntries))

        #for i in tqdm(range(0, len(racecodeEntries))):
        for i in range(0, len(racecodeEntries)):

            entry = racecodeEntries[targetColumnString][i].lower()

            #print(entry)
            
            res = 'other_ethnic_group'

            count = 0

            for synonym in whiteList:   
                if(synonym in entry and entry not in set(allEthList).difference(set(whiteList))):   
                    count = count + 1
                    res = 'white'
                    #print(entry, 'white')


            for synonym in asianList:
                if(synonym in entry and entry not in set(allEthList).difference(set(asianList))):   
                    count = count + 1
                    res = 'asian_or_asian_british'    


            for synonym in blackList:
                if(synonym in entry and entry not in set(allEthList).difference(set(blackList))):
                    count = count + 1
                    res = 'black_african_caribbean_or_black_british'


            for synonym in otherList:    
                if(synonym in entry and entry not in set(allEthList).difference(set(otherList))):  
                    count = count + 1
                    res = 'other_ethnic_group' 
                    #print(entry, 'other_ethnic_group2')


            for synonym in mixedList:    
                if(synonym in entry and entry not in set(allEthList).difference(set(mixedList))):    
                    count = count + 1
                    res = 'mixed_or_multiple_ethnic_groups'


            #Explicit/specification:
            if('other' in entry and entry not in set(allEthList).difference(set(otherList))):
                res = 'other_ethnic_group'
                #print(entry, 'other_ethnic_group')


            if('black' in entry and entry not in set(allEthList).difference(set(blackList))):
                res = 'black_african_caribbean_or_black_british'

            if('white' in entry and entry not in set(allEthList).difference(set(whiteList))):
                res = 'white'

            if('asian' in entry and entry not in set(allEthList).difference(set(asianList))):
                res = 'asian_or_asian_british'


            if('mix' in entry and entry not in set(allEthList).difference(set(mixedList))):
                res = 'mixed_or_multiple_ethnic_groups'





            if count> 15:
                #print("Mixed found:")
                #print(entry)
                #print(entry)
                #res = 'Mixed or Multiple ethnic groups'
                pass

            #print("Returning ", res)
            df_testMap['census'][i] = res

            #print(df_testMap['census'].str.count("Other ethnic group").sum(), len(racecodeEntries))
        return df_testMap



##Awaiting integration

# add_filter = False

# filter_root_term = 'infection'

# filter_root_cui = '40733004'
import requests


def get_snowstorm_response_children(cui):
    url = f"https://snowstorm.ihtsdotools.org/snowstorm/snomed-ct/browser/MAIN%2FSNOMEDCT-GB/concepts/{cui}/children?form=inferred&includeDescendantCount=false"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
        # Add other headers if required
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    
    
# if(add_filter):
#     gssr_result = json.loads(get_snowstorm_response_children(filter_root_cui))
#     len(gssr_result),gssr_result[0], gssr_result[0]['id'],gssr_result[0]['fsn']['term'] 


#     cui_filter_list = []
#     name_filter_list = []
#     for i in range(0, len(gssr_result)):
#         cui_filter_list.append(gssr_result[i]['id'])
#         name_filter_list.append(gssr_result[i]['fsn']['term'])

#     print(len(cui_filter_list), cui_filter_list[0:3])

# if(add_filter):

#     cat.cdb.filter_by_cui(cui_filter_list)
#     print("added filter")


def format_string_list(input_list):
    if not input_list:
        return ""

    formatted_string = ""
    for item in input_list:
        formatted_string += f'OR "{item}"\n'

    return formatted_string.rstrip("\n")