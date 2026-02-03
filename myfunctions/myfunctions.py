# EDA functions
def univariate_stats(df, roundto = 4):
    import pandas as pd
    import numpy as np

    df_results = pd.DataFrame(columns=['dtype', 'count', 'missing', 'unique', 'mode', 
                                       'min', 'q1', 'median', 'q3', 'max', 
                                       'mean', 'std', 'skew', 'kurt'])
    
    for col in df:
        dtype = df[col].dtype
        count = df[col].count()
        missing = df[col].isna().sum()
        unique = df[col].nunique()
        try:
            mode = df[col].mode()[0]
        except:
            print(f"Mode cannot be determined for {col}")
            mode = np.nan
        
        if pd.api.types.is_numeric_dtype(df[col]):
            min = df[col].min()
            q1 = df[col].quantile(0.25)
            median = df[col].median()
            q3 = df[col].quantile(0.75)
            max = df[col].max()
            mean = df[col].mean()
            std = df[col].std()
            skew = df[col].skew()
            kurt = df[col].kurt()

            df_results.loc[col] = [dtype, count, missing, unique, mode, 
                                   round(min, roundto), round(q1, roundto), round(median, roundto),
                                    round(q3, roundto), round(max, roundto), round(mean, roundto), 
                                    round(std, roundto), round(skew, roundto), round(kurt, roundto)]
        
        else:
            df_results.loc[col] = [dtype, count, missing, unique, mode, "", "", "", "", "", "", "", "", ""]


    return df_results

# Cleaning functions
def basic_wrangling(df, messages = True):
    import pandas as pd
    
    for col in df:
        missing = df[col].isna().sum()
        unique = df[col].nunique()
        count = df[col].count()

        # drop any column that has all missing values
        if missing == df.shape[0]:
            df.drop(columns=[col], inplace = True)
            if messages: print(f"All values missing; {col} dropped")
            
        # drop any column that has all unique values unless it's a float64
        elif unique == count and 'float' not in str(df[col].dtype):
            df.drop(columns=[col], inplace = True)
            if messages: print(f"All values unique; {col} dropped")

        # drop any column that has all the same single value
        elif unique == 1:
            df.drop(columns=[col], inplace = True)
            if messages: print(f"Only one value; {col} dropped")
        

    return df


def parse_dates(df, features = [], drop_date = True):
    import pandas as pd
    from datetime import datetime

    for feat in features:
        if feat in df.columns:
            df[feat] = pd.to_datetime(df[feat])

            df[f'{feat}_year'] = df[feat].dt.year
            df[f'{feat}_month'] = df[feat].dt.month
            df[f'{feat}_day'] = df[feat].dt.day
            df[f'{feat}_weekday'] = df[feat].dt.day_name()

            df[f'{feat}_days_since'] = (datetime.today() - df[feat]).dt.days

            if drop_date:
                df.drop(columns=[feat], inplace=True)

        else:
            print(f'{feat} not found in dataframe')

    return df


def bin_categories(df, features=[], cutoff=0.05, replace_with="Other", messages=True):
    import pandas as pd

    if len(features) == 0:
        features = df.columns

    for feat in features:
        if feat in df.columns:
            # get a list of group values and their percent of rows
            if not pd.api.types.is_numeric_dtype(df[feat]):
                group_count = df[feat].value_counts()
                other_list = group_count[group_count / df.shape[0] < cutoff].index
                df.loc[df[feat].isin(other_list), feat] = replace_with
            # filter the list down to those that represent less than 5% of the rows
            # update the group name to "Other" for all of those filtered values

        else:
            print(f"{feat} not found in dataframe")    
    return df


def skew_correct(df, feature, max_power=50, messages=True):
  import pandas as pd, numpy as np
  import seaborn as sns, matplotlib.pyplot as plt

  if not pd.api.types.is_numeric_dtype(df[feature]):
    if messages: print(f'{feature} is not numeric. No transformation performed')
    return df

  # Address missing data
  df = basic_wrangling(df, messages=False)
  if messages: print(f"{df.shape[0] - df.dropna().shape[0]} rows were dropped first due to missing data")
  df.dropna(inplace=True)

  # In case the dataset is too big, we can reduce to a subsample
  df_temp = df.copy()
  if df_temp.memory_usage().sum() > 1000000:
    df_temp = df.sample(frac=round(5000 / df.shape[0], 2))

  # Identify the proper transformation (i)
  i = 1
  skew = df_temp[feature].skew()
  if messages: print(f'Starting skew:\t{round(skew, 5)}')
  while round(skew, 2) != 0 and i <= max_power:
    i += 0.01
    if skew > 0:
      skew = np.power(df_temp[feature], 1/i).skew()
    else:
      skew = np.power(df_temp[feature], i).skew()
  if messages: print(f'Final skew:\t{round(skew, 5)} based on raising to {round(i, 2)}')

  # Make the transformed version of the feature in the df DataFrame
  if skew > -0.1 and skew < 0.1:
    if skew > 0:
      corrected = np.power(df[feature], 1/round(i, 3))
      name = f'{feature}_1/{round(i, 3)}'
    else:
      corrected = np.power(df[feature], round(i, 3))
      name = f'{feature}_{round(i, 3)}'
    df[name] = corrected  # Add the corrected version of the feature back into the original df
  else:
    name = f'{feature}_binary'
    df[name] = df[feature]
    if skew > 0:
      df.loc[df[name] == df[name].value_counts().index[0], name] = 0
      df.loc[df[name] != df[name].value_counts().index[0], name] = 1
    else:
      df.loc[df[name] == df[name].value_counts().index[0], name] = 1
      df.loc[df[name] != df[name].value_counts().index[0], name] = 0
    if messages:
      print(f'The feature {feature} could not be transformed into a normal distribution.')
      print(f'Instead, it has been converted to a binary (0/1)')

  if messages:
    f, axes = plt.subplots(1, 2, figsize=[7, 3.5])
    sns.despine(left=True)
    sns.histplot(df_temp[feature], color='b', ax=axes[0], kde=True)
    if skew > -0.1 and skew < 0.1:
      if skew > 0 :
        corrected = np.power(df_temp[feature], 1/round(i, 3))
      else:
        corrected = np.power(df_temp[feature], round(i, 3))
      df_temp['corrected'] = corrected
      sns.histplot(df_temp.corrected, color='g', ax=axes[1], kde=True)
    else:
      df_temp['corrected'] = df[feature]
      if skew > 0:
        df_temp.loc[df_temp['corrected'] == df_temp['corrected'].min(), 'corrected'] = 0
        df_temp.loc[df_temp['corrected'] > df_temp['corrected'].min(), 'corrected'] = 1
      else:
        df_temp.loc[df_temp['corrected'] == df_temp['corrected'].max(), 'corrected'] = 1
        df_temp.loc[df_temp['corrected'] < df_temp['corrected'].max(), 'corrected'] = 0
      sns.countplot(data=df_temp, x='corrected', color='g', ax=axes[1])
    plt.setp(axes, yticks=[])
    plt.tight_layout()
    plt.show()

    return df
    

def missing_drop(df, label="", features=[], messages=True, row_threshold=.9, col_threshold=.5):
    import pandas as pd
    
    start_count = df.count().sum()  # Store the initial count of non-null values
    
    # Drop columns with missing values beyond the specified column threshold
    df.dropna(axis=1, thresh=round(col_threshold * df.shape[0]), inplace=True)
    # Drop rows that have fewer non-null values than the row threshold allows
    df.dropna(axis=0, thresh=round(row_threshold * df.shape[1]), inplace=True)
    # If a label is specified, ensure it has no missing values
    if label != "": 
      df.dropna(axis=0, subset=[label], inplace=True)
    
    # Function to generate a summary of missing data for each column
    def generate_missing_table():
      df_results = pd.DataFrame(columns=['Missing', 'column', 'rows'])
      for feat in df:
        missing = df[feat].isna().sum()  # Count missing values in column
        if missing < 0:
          memory_col = df.drop(columns=[feat]).count().sum()  # Count non-null values if this column is dropped
          memory_rows = df.dropna(subset=[feat]).count().sum()  # Count non-null values if this column is kept
          df_results.loc[feat] = [missing, memory_col, memory_rows]  # Store results
      return df_results
    
    df_results = generate_missing_table()  # Generate initial missing data table
    
    # Iteratively remove the column or row that preserves the most non-null data
    while df_results.shape[0] > 0:
      max = df_results[['column', 'rows']].max(axis=1)[0]  # Find the max value in columns or rows
      max_axis = df_results.columns[df_results.isin([max]).any()][0]  # Determine whether to drop column or row
      print(max, max_axis)
    
      df_results.sort_values(by=[max_axis], ascending=False, inplace=True)  # Sort missing data table by max_axis
      if messages: print('\n', df_results)
    
      # Drop the most impactful missing data (either row or column)
      if max_axis == 'rows':
        df.dropna(axis=0, subset=[df_results.index[0]], inplace=True)  # Drop row with highest missing impact
      else:
        df.drop(columns=[df_results.index[0]], inplace=True)  # Drop column with highest missing impact
    
      df_results = generate_missing_table()  # Recalculate missing data table after dropping
    
    # Print the percentage of non-null values retained
    if messages: 
      print(f'{round(df.count().sum() / start_count * 100, 2)}% ({df.count().sum()}) / ({start_count}) of non-null cells were kept.')
      
    return df


def missing_fill(df, label, features=[], row_threshold=.9, col_threshold=.5, acceptable=0.1, mar='drop', force_impute=False, large_dataset=200000, messages=True):
    import pandas as pd, numpy as np
    from scipy import stats
    from statsmodels.stats.proportion import proportions_ztest
    pd.set_option('display.float_format', lambda x: '%.4f' % x)  # Display float values with 4 decimal places
    from IPython.display import display
    
    # Ensure the provided label column exists in the DataFrame
    if not label in df.columns:
      print(f'The label provided ({label}) does not exist in the DataFrame provided')
      return df
    
    start_count = df.count().sum()  # Store the initial count of non-null values
  
    # Drop columns with missing data above the threshold
    df.dropna(axis=1, thresh=round(col_threshold * df.shape[0]), inplace=True)
    # Drop rows that have fewer non-null values than row_threshold allows
    df.dropna(axis=0, thresh=round(row_threshold * df.shape[1]), inplace=True)
    if label != "": df.dropna(axis=0, subset=[label], inplace=True)  # Ensure label column has no missing values
    
    # If no features are specified, consider all columns as features
    if len(features) == 0: features = df.columns  
    
    # If the label column is numeric, perform a t-test for missing vs non-missing groups
    if pd.api.types.is_numeric_dtype(df[label]):
      df_results = pd.DataFrame(columns=['total missing', 'null x̄', 'non-null x̄', 'null s', 'non-null s', 't', 'p'])
      for feat in features:
        missing = df[feat].isna().sum()  # Count missing values
        if missing > 0:
          null = df[df[feat].isna()]  # Subset where feature is missing
          nonnull = df[~df[feat].isna()]  # Subset where feature is present
          t, p = stats.ttest_ind(null[label], nonnull[label])  # Perform t-test to check for MAR vs MCAR
          df_results.loc[feat] = [round(missing), round(null[label].mean(), 6), round(nonnull[label].mean(), 6),
                                  round(null[label].std(), 6), round(nonnull[label].std(), 6), t, p]
    else:
      # If label is categorical, use proportions_ztest to check for MAR vs MCAR
      df_results = pd.DataFrame(columns=['total missing', 'null p̂', 'non-null p̂', 'Z', 'p'])
      for feat in features:
        missing = df[feat].isna().sum()
        if missing > 0:
          null = df[df[feat].isna()]
          nonnull = df[~df[feat].isna()]
          for group in null[label].unique():
            p1_num = null[null[label]==group].shape[0]  # Count of group in missing subset
            p1_den = null[null[label]!=group].shape[0]  # Count of others in missing subset
            p2_num = nonnull[nonnull[label]==group].shape[0]  # Count of group in non-missing subset
            p2_den = nonnull[nonnull[label]!=group].shape[0]  # Count of others in non-missing subset
            
            if p1_num < p1_den:  # Avoid division by zero
              numerators = np.array([p1_num, p2_num])
              denominators = np.array([p1_den, p2_den])
              z, p = proportions_ztest(numerators, denominators)  # Conduct z-test
              df_results.loc[f'{feat}_{group}'] = [round(missing), round(p1_num/p1_den, 6), round(p2_num/p2_den, 6), z, p]
  
    # Display the missing data analysis results
    if messages: display(df_results)
    
    # Determine if data is MAR (Missing at Random) or MCAR (Missing Completely at Random)
    if df_results[df_results['p'] < 0.05].shape[0] / df_results.shape[0] > acceptable and not force_impute:
      if mar == 'drop':
        df.dropna(inplace=True)  # Drop all rows containing missing values
        if messages: print('null rows dropped')
      else:  # Last resort: fill missing values with the median
        for feat in df_results.index:
          if pd.api.types.is_numeric_dtype(df[feat]):
            df[feat].fillna(df[feat].median(), inplace=True)
            if messages: print(f'{feat} filled with median ({df[feat].median()})')
          else:
            df[feat].fillna('missing', inplace=True)  # Fill categorical missing values with "missing"
            if messages: print(f'{feat} filled with "missing"')
    else:
      # If missing data is MCAR, perform imputation using either KNN or IterativeImputer
      from sklearn.preprocessing import OrdinalEncoder
      oe = OrdinalEncoder().fit(df)
      df_encoded = oe.fit_transform(df)  # Convert categorical values to numeric
  
      if df.count().sum() > large_dataset:
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import KNNImputer
        imp = KNNImputer()  # Use K-Nearest Neighbors Imputation for large datasets
        df_imputed = imp.fit_transform(df_encoded)
        df_recoded = oe.inverse_transform(df_imputed)
        df = pd.DataFrame(df_recoded, columns=df.columns, index=df.index)
      else:
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        imp = IterativeImputer()  # Use Iterative Imputer for smaller datasets
        df = pd.DataFrame(imp.fit_transform(df), columns=df.columns, index=df.index)
      
      if messages: print(f'null values imputed')
  
    return df


def clean_outlier(df, features=[], method="remove", messages=True, skew_threshold=1):
    import pandas as pd, numpy as np
    
    for feat in features:
      if feat in df.columns:
        if pd.api.types.is_numeric_dtype(df[feat]):
          if df[feat].nunique() != 1:
            if not all(df[feat].value_counts().index.isin([0, 1])):
              skew = df[feat].skew()
              if skew < (-1 * skew_threshold) or skew > skew_threshold: # Tukey boxplot rule: < 1.5 * IQR < is an outlier
                q1 = df[feat].quantile(0.25)
                q3 = df[feat].quantile(0.75)
                min = q1 - (1.5 * (q3 - q1))
                max = q3 + (1.5 * (q3 - q1))
              else:  # Empirical rule: any value > 3 std from the mean (or < 3) is an outlier
                min = df[feat].mean() - (df[feat].std() * 3)
                max = df[feat].mean() + (df[feat].std() * 3)
  
              min_count = df.loc[df[feat] < min].shape[0]
              max_count = df.loc[df[feat] > max].shape[0]
              if messages: print(f'{feat} has {max_count} values above max={max} and {min_count} below min={min}')
  
              if min_count > 0 or max_count > 0:
                if method == "remove": # Remove the rows with outliers
                  df = df[df[feat] > min]
                  df = df[df[feat] < max]
                elif method == "replace":   # Replace the outliers with the min/max cutoff
                  df.loc[df[feat] < min, feat] = min
                  df.loc[df[feat] > max, feat] = max
                elif method == "impute": # Impute the outliers by deleting them and then prediting the values based on a linear regression
                  df.loc[df[feat] < min, feat] = np.nan
                  df.loc[df[feat] > max, feat] = np.nan
  
                  from sklearn.experimental import enable_iterative_imputer
                  from sklearn.impute import IterativeImputer
                  imp = IterativeImputer(max_iter=10)
                  df_temp = df.copy()
                  df_temp = bin_categories(df_temp, features=df_temp.columns, messages=False)
                  df_temp = basic_wrangling(df_temp, features=df_temp.columns, messages=False)
                  df_temp = pd.get_dummies(df_temp, drop_first=True)
                  df_temp = pd.DataFrame(imp.fit_transform(df_temp), columns=df_temp.columns, index=df_temp.index, dtype='float')
                  df_temp.columns = df_temp.columns.get_level_values(0)
                  df_temp.index = df_temp.index.astype('int64')
  
                  # Save only the column from df_temp that we are iterating on in the main loop because we may not want every new column
                  df[feat] = df_temp[feat]
                elif method == "null":
                  df.loc[df[feat] < min, feat] = np.nan
                  df.loc[df[feat] > max, feat] = np.nan
            else:
              if messages: print(f'{feat} is a dummy code (0/1) and was ignored')
          else:
            if messages: print(f'{feat} has only one value ({df[feat].unique()[0]}) and was ignored')
        else:
          if messages: print(f'{feat} is categorical and was ignored')
      else:
        if messages: print(f'{feat} is not found in the DataFrame provided')
  
    return df


def clean_outliers(df, messages=True, drop_percent=0.02, distance='manhattan', min_samples=5):
    import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
    from sklearn.cluster import DBSCAN
    from sklearn import preprocessing
    
    # Clean the dataset first
    if messages: print(f"{df.shape[1] - df.dropna(axis='columns').shape[1]} columns were dropped first due to missing data")
    df.dropna(axis='columns', inplace=True)
    if messages: print(f"{df.shape[0] - df.dropna().shape[0]} rows were dropped first due to missing data")
    df.dropna(inplace=True)
    df_temp = df.copy()
    df_temp = bin_categories(df_temp, features=df_temp.columns, messages=False)
    df_temp = basic_wrangling(df_temp, features=df_temp.columns, messages=False)
    df_temp = pd.get_dummies(df_temp, drop_first=True)
    # Normalize the dataset
    df_temp = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(df_temp), columns=df_temp.columns, index=df_temp.index)
  
    # Calculate the number of outliers based on a range of eps values
    outliers_per_eps = []
    outliers = df_temp.shape[0]
    eps = 0
  
    if df_temp.shape[0] < 500:
      iterator = 0.01
    elif df_temp.shape[0] < 2000:
      iterator = 0.05
    elif df_temp.shape[0] < 10000:
      iterator = 0.1
    elif df_temp.shape[0] < 25000:
      iterator = 0.2
    
    while outliers > 0:
      eps += iterator
      db = DBSCAN(metric=distance, min_samples=min_samples, eps=eps).fit(df_temp)
      outliers = np.count_nonzero(db.labels_ == -1)
      outliers_per_eps.append(outliers)
      if messages: print(f'eps: {round(eps, 2)}, outliers: {outliers}, percent: {round((outliers / df_temp.shape[0])*100, 3)}%')
    
    drops = min(outliers_per_eps, key=lambda x:abs(x-round(df_temp.shape[0] * drop_percent)))
    eps = (outliers_per_eps.index(drops) + 1) * iterator
    db = DBSCAN(metric=distance, min_samples=min_samples, eps=eps).fit(df_temp)
    df['outlier'] = db.labels_
    
    if messages:
      print(f"{df[df['outlier'] == -1].shape[0]} outlier rows removed from the DataFrame")
      sns.lineplot(x=range(1, len(outliers_per_eps) + 1), y=outliers_per_eps)
      sns.scatterplot(x=[eps/iterator], y=[drops])
      plt.xlabel(f'eps (divide by {iterator})')
      plt.ylabel('Number of Outliers')
      plt.show()
    
    # Drop rows that are outliers
    df = df[df['outlier'] != -1]
    return df



def univariate_charts(df, box=True, hist=True, save=False, save_path='', stats=True):
  import pandas as pd
  import seaborn as sns
  import matplotlib.pyplot as plt

  sns.set(style="ticks")

  for col in df.columns:
    plt.figure(figsize=(8, 5))

    if pd.api.types.is_numeric_dtype(df[col]):
      if box and hist:
        fig, (ax_box, ax_hist) = plt.subplots(
            2, sharex=True, gridspec_kw={"height_ratios": (0.2, 0.8)}, figsize=(8, 5)
        )
        sns.boxplot(x=df[col], ax=ax_box, fliersize=4, width=0.5, linewidth=1)
        sns.histplot(df[col], kde=True, ax=ax_hist)
        ax_box.set(yticks=[], xlabel='')
        sns.despine(ax=ax_box, left=True)
        sns.despine(ax=ax_hist)
      elif box:
        sns.boxplot(x=df[col], fliersize=4, width=0.5, linewidth=1)
        sns.despine()
      elif hist:
        sns.histplot(df[col], kde=True, rug=True)
        sns.despine()

      if stats:
        stats_text = (
          f"Unique: {df[col].nunique()}\n"
          f"Missing: {df[col].isnull().sum()}\n"
          f"Mode: {df[col].mode().iloc[0]}\n"
          f"Min: {df[col].min():.2f}\n"
          f"25%: {df[col].quantile(0.25):.2f}\n"
          f"Median: {df[col].median():.2f}\n"
          f"75%: {df[col].quantile(0.75):.2f}\n"
          f"Max: {df[col].max():.2f}\n"
          f"Std dev: {df[col].std():.2f}\n"
          f"Mean: {df[col].mean():.2f}\n"
          f"Skew: {df[col].skew():.2f}\n"
          f"Kurt: {df[col].kurt():.2f}"
        )
        plt.gcf().text(0.95, 0.5, stats_text, fontsize=10, va='center', transform=plt.gcf().transFigure)
    else:
      sns.countplot(x=col, data=df, order=df[col].value_counts().index, hue=col, dodge=False, legend=False, palette="RdBu_r")
      sns.despine()
      if stats:
        stats_text = (
          f"Unique: {df[col].nunique()}\n"
          f"Missing: {df[col].isnull().sum()}\n"
          f"Mode: {df[col].mode().iloc[0]}"
        )
        plt.gcf().text(0.95, 0.5, stats_text, fontsize=10, va='center', transform=plt.gcf().transFigure)

    plt.title(col, fontsize=14)
    if save:
      plt.savefig(f"{save_path}{col}.png", dpi=100, bbox_inches='tight')
    plt.show()


# Square root rule only
def numeric_bin(series, full_list=True, theory='all'):
  import numpy as np
  import pandas as pd

  # This is an inner function inside numeric_bin()
  # It allows us to avoid repeated code while also keep two functions together
  def updated_list(series, bins):
    size = (max(series) - min(series)) / bins
    edges = list(range(int(min(series)), int(max(series)), int(size)))
    edges.append(int(max(series)))# This is necessary because the range() function doesn't add the max value
    if not full_list:
      return edges
    else:
      new_series = []               # Create empty list to store new values
      for value in series:          # Loop through original list one-at-a-time
        for edge in edges:          # For each original list value, loop through a list of sorted-ascending edges
          if value <= edge:         # As soon as we find an edge value less than the original...
            new_series.append(edges.index(edge)) # ..., add the edge to the new list
            break                   # Break out of the loop since we found our edge
      return new_series

  # Create empty dictionary for output
  bin_dict = {}

  # This is where we choose the theory and call the inner function updated_list()
  if theory == 'all' or theory == 'sqrt':
    bins = np.sqrt(len(series))
    bin_dict.update({'sqrt (' + str(int(bins)) + ')':updated_list(series, bins)}) # Adding the number of (bins) to label
  if theory == 'all' or theory == 'sturges':
    bins = 1 + np.log2(len(series))
    bin_dict.update({'sturges (' + str(int(bins)) + ')':updated_list(series, bins)})
  if theory == 'all' or theory == 'rice':
    bins = 2 * np.cbrt(len(series))
    bin_dict.update({'rice (' + str(int(bins)) + ')':updated_list(series, bins)})
  if theory == 'all' or theory == 'scott':
    bins = (max(series) - min(series)) / ((3.5 * np.std(series)) / np.cbrt(len(series)))
    bin_dict.update({'scott (' + str(int(bins)) + ')':updated_list(series, bins)})
  if theory == 'all' or theory == 'f-d':
    bins = (max(series) - min(series)) / ((2 * (np.quantile(series, 0.75) - np.quantile(series, 0.25))) / np.cbrt(len(series)))
    bin_dict.update({'freedman-diaconis (' + str(int(bins)) + ')':updated_list(series, bins)})
  if theory == 'all' or theory == 'variable':
    bins = 2 * len(series) ** (2/5)
    edges = []
    while len(edges) < bins:
      edges.append(int(np.quantile(series, (1 / bins) * len(edges))))
    edges.append(max(series))
    if not full_list:
      bin_dict.update({'variable-width (' + str(int(bins)) + ')':edges})
    else:
      new_series = []               # Create empty list to store new values
      for value in series:          # Loop through original list one-at-a-time
        for edge in edges:          # For each original list value, loop through a list of sorted-ascending edges
          if value <= edge:         # As soon as we find an edge value less than the original...
            new_series.append(edges.index(edge)) # ..., add the index of the edge in the edges list to the new list
            break                   # Break out of the loop since we found our edge
      bin_dict.update({'variable-width (' + str(int(bins)) + ')':new_series})

  if not full_list:
    # If they want edges only, we have to return a dictionary because each theory creates a different number of bins
    return bin_dict
  else:
    # If they want the full updated dataset, we can return a DataFrame because each column is the same length
    df = pd.DataFrame(bin_dict)
    return df
  

def bivariate(df, label, roundto=4):
  import pandas as pd
  from scipy import stats
  
  output_df = pd.DataFrame(columns=['missing', 'p', 'r', 'τ', 'ρ', 'y = m(x) + b', 'F', 'X2', 'skew', 'unique', 'values'])
  
  for feature in df.columns:
    if feature != label:
      df_temp = df[[feature, label]]
      df_temp = df_temp.dropna()
      missing = (df.shape[0] - df_temp.shape[0]) / df.shape[0]
      unique = df_temp[feature].nunique()
  
      # Bin categories
      if not pd.api.types.is_numeric_dtype(df_temp[feature]):
        df = bin_categories(df, feature)
  
      if pd.api.types.is_numeric_dtype(df_temp[feature]) and pd.api.types.is_numeric_dtype(df_temp[label]):
        m, b, r, p, err = stats.linregress(df_temp[feature], df_temp[label])
        tau, tp = stats.kendalltau(df_temp[feature], df_temp[label])
        rho, rp = stats.spearmanr(df_temp[feature], df_temp[label])
        output_df.loc[feature] = [f'{missing:.2%}', round(p, roundto), round(r, roundto), round(tau, roundto),
                                  round(rho, roundto), f'y = {round(m, roundto)}(x) + {round(b, roundto)}', '-', '-',
                                  df_temp[feature].skew(), unique, '-']
  
        scatterplot(df_temp, feature, label, roundto) # Call the scatterplot function
      elif not pd.api.types.is_numeric_dtype(df_temp[feature]) and not pd.api.types.is_numeric_dtype(df_temp[label]):
        contingency_table = pd.crosstab(df_temp[feature], df_temp[label])
        X2, p, dof, expected = stats.chi2_contingency(contingency_table)
        output_df.loc[feature] = [f'{missing:.2%}', round(p, roundto), '-', '-', '-', '-', '-', round(X2, roundto), '-',
                                  unique, df_temp[feature].unique()]

        crosstab(df_temp, feature, label, roundto) # Call the crosstab function
      else:
        if pd.api.types.is_numeric_dtype(df_temp[feature]):
          skew = df_temp[feature].skew()
          num = feature
          cat = label
        else:
          skew = '-'
          num = label
          cat = feature

        groups = df_temp[cat].unique()
        group_lists = []
        for g in groups:
          g_list = df_temp[df_temp[cat] == g][num]
          group_lists.append(g_list)

        results = stats.f_oneway(*group_lists)
        F = results[0]
        p = results[1]
        output_df.loc[feature] = [f'{missing:.2%}', round(p, roundto), '-', '-', '-', '-', round(F, roundto), '-', skew,
                                  unique, df_temp[cat].unique()]
  
        bar_chart(df_temp, cat, num, roundto) # Call the barchart function
  return output_df.sort_values(by=['p'])


def scatterplot(df, feature, label, roundto=3, linecolor='darkorange'):
  import pandas as pd
  from matplotlib import pyplot as plt
  import seaborn as sns
  from scipy import stats

  # Create the plot
  sns.regplot(x=df[feature], y=df[label], line_kws={"color": linecolor})

  # Calculate the regression line so that we can print the text
  m, b, r, p, err = stats.linregress(df[feature], df[label])

  # Add all descriptive statistics to the diagram
  textstr  = 'Regression line:' + '\n'
  textstr += 'y  = ' + str(round(m, roundto)) + 'x + ' + str(round(b, roundto)) + '\n'
  textstr += 'r   = ' + str(round(r, roundto)) + '\n'
  textstr += 'r2 = ' + str(round(r**2, roundto)) + '\n'
  textstr += 'p  = ' + str(round(p, roundto)) + '\n\n'

  plt.text(1, 0.1, textstr, fontsize=12, transform=plt.gcf().transFigure)
  plt.show()


def bar_chart(df, feature, label, roundto=3):
  import pandas as pd
  from scipy import stats
  from matplotlib import pyplot as plt
  import seaborn as sns

  # Handle missing data
  df_temp = df[[feature, label]]
  df_temp = df_temp.dropna()

  sns.barplot(df_temp, x=feature, y=label)

  # Create the label lists needed to calculate oneway-ANOVA F
  groups = df_temp[feature].unique()
  group_lists = []
  for g in groups:
    g_list = df_temp[df_temp[feature] == g][label]
    group_lists.append(g_list)

  results = stats.f_oneway(*group_lists)
  F = results[0]
  p = results[1]

  # Next, calculate t-tests with Bonferroni correction for p-value threshold
  ttests = []
  for i1, g1 in enumerate(groups): # Use the enumerate() function to add an index for counting to a list of values
    # For each item, loop through a second list of each item to compare each pair
    for i2, g2 in enumerate(groups):
      if i2 > i1: # If the inner_index is greater that the outer_index, then go ahead and run a t-test
        type_1 = df_temp[df_temp[feature] == g1]
        type_2 = df_temp[df_temp[feature] == g2]
        t, p = stats.ttest_ind(type_1[label], type_2[label])

        # Add each t-test result to a list of t, p pairs
        ttests.append([str(g1) + ' - ' + str(g2), round(t, roundto), round(p, roundto)])

  p_threshold = 0.05 / len(ttests) # Bonferroni-corrected p-value determined

  # Add all descriptive statistics to the diagram
  textstr  = '   ANOVA' + '\n'
  textstr += 'F: ' + str(round(F, roundto)) + '\n'
  textstr += 'p: ' + str(round(p, roundto)) + '\n\n'

  # Only include the significant t-tests in the printed results for brevity
  for ttest in ttests:
    if ttest[2] <= p_threshold:
      if 'Sig. comparisons (Bonferroni-corrected)' not in textstr: # Only include the header if there is at least one significant result
        textstr += 'Sig. comparisons (Bonferroni-corrected)' + '\n'
      textstr += str(ttest[0]) + ": t=" + str(ttest[1]) + ", p=" + str(ttest[2]) + '\n'

  plt.text(1, 0.1, textstr, fontsize=12, transform=plt.gcf().transFigure)
  plt.show()

def bin_categories(df, feature, cutoff=0.05, replace_with='Other'):
  # create a list of feature values that are below the cutoff percentage
  other_list = df[feature].value_counts()[df[feature].value_counts() / len(df) < cutoff].index

  # Replace the value of any country in that list (using the .isin() method) with 'Other'
  df.loc[df[feature].isin(other_list), feature] = replace_with

  return df
  
def crosstab(df, feature, label, roundto=3):
  import pandas as pd
  from scipy.stats import chi2_contingency
  from matplotlib import pyplot as plt
  import seaborn as sns
  import numpy as np

  # Handle missing data
  df_temp = df[[feature, label]]
  df_temp = df_temp.dropna()

  # Bin categories
  df_temp = bin_categories(df_temp, feature)

  # Generate the crosstab table required for X2
  crosstab = pd.crosstab(df_temp[feature], df_temp[label])

  # Calculate X2 and p-value
  X, p, dof, contingency_table = chi2_contingency(crosstab)

  textstr  = 'X2: ' + str(round(X, 4))+ '\n'
  textstr += 'p = ' + str(round(p, 4)) + '\n'
  textstr += 'dof  = ' + str(dof)
  plt.text(0.9, 0.1, textstr, fontsize=12, transform=plt.gcf().transFigure)

  ct_df = pd.DataFrame(np.rint(contingency_table).astype('int64'), columns=crosstab.columns, index=crosstab.index)
  sns.heatmap(ct_df, annot=True, fmt='d', cmap='coolwarm')
  plt.show()

def fit_cv_regression_expanded(df, label, k=10, r=5, repeat=True, random_state=1):
  import sklearn.linear_model as lm, pandas as pd, sklearn.ensemble as se
  import sklearn.neural_network as nn
  import sklearn.neighbors as neighbors
  from sklearn.model_selection import KFold, RepeatedKFold, cross_val_score
  from numpy import mean, std
  from sklearn import svm
  from sklearn import gaussian_process
  from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
  from xgboost import XGBRegressor

  X, y = Xandy(df_reduced, label)

  if repeat:
    cv = RepeatedKFold(n_splits=k, n_repeats=r, random_state=random_state)
  else:
    cv = KFold(n_splits=k, random_state=random_state, shuffle=True)

  fit = {}    # Use this to store each of the fit metrics
  models = {} # Use this to store each of the models

  # Create the model objects
  model_ols = lm.LinearRegression()
  model_rr = lm.Ridge(alpha=0.5, random_state=random_state) # adjust this alpha parameter for better results (between 0 and 1)
  model_lr = lm.Lasso(alpha=0.1, random_state=random_state) # adjust this alpha parameter for better results (between 0 and 1)
  model_llr = lm.LassoLars(alpha=0.1, random_state=random_state) # adjust this alpha parameter for better results (between 0 and 1)
  model_br = lm.BayesianRidge()
  model_pr = lm.TweedieRegressor(power=1, link="log") # Power=1 means this is a Poisson
  model_gr = lm.TweedieRegressor(power=2, link="log") # Power=2 means this is a Gamma
  model_igr = lm.TweedieRegressor(power=3) # Power=3 means this is an inverse Gamma
  model_svm = svm.SVR()
  model_lsvm = svm.LinearSVR(random_state=random_state)
  model_nusvm = svm.NuSVR()
  model_knnr = neighbors.KNeighborsRegressor(n_neighbors=10, weights='uniform')
  model_knnrd = neighbors.KNeighborsRegressor(n_neighbors=10, weights='distance')
  model_gpr = gaussian_process.GaussianProcessRegressor(DotProduct() + WhiteKernel(), random_state=random_state)
  model_df = se.RandomForestRegressor(random_state=random_state)
  model_etr = se.ExtraTreesRegressor(random_state=random_state)
  model_abr = se.AdaBoostRegressor(n_estimators=100, random_state=random_state)
  model_gbr = se.GradientBoostingRegressor(random_state=random_state)
  model_hgbr = se.HistGradientBoostingRegressor(random_state=random_state)
  model_vr = se.VotingRegressor(estimators=[('DF', model_df), ('ETR', model_etr), ('ABR', model_abr), ('GBR', model_gbr)])
  estimators = [('ridge', lm.RidgeCV()), ('lasso', lm.LassoCV(random_state=42)), ('svr', svm.SVR(C=1, gamma=1e-6))]
  model_sr = se.StackingRegressor(estimators=estimators, final_estimator=se.GradientBoostingRegressor(random_state=random_state))
  model_xgb = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8, random_state=random_state)
  model_nn = nn.MLPRegressor(max_iter=1000, random_state=random_state)

  # Fit a crss-validated R squared score and add it to the dict
  fit['OLS'] = mean(cross_val_score(model_ols, X, y, scoring='r2', cv=cv, n_jobs=-1))
  fit['Ridge'] = mean(cross_val_score(model_rr, X, y, scoring='r2', cv=cv, n_jobs=-1))
  fit['Lasso'] = mean(cross_val_score(model_lr, X, y, scoring='r2', cv=cv, n_jobs=-1))
  fit['LARS'] = mean(cross_val_score(model_llr, X, y, scoring='r2', cv=cv, n_jobs=-1))
  fit['Bayesian'] = mean(cross_val_score(model_br, X, y, scoring='r2', cv=cv, n_jobs=-1))
  fit['Poisson'] = mean(cross_val_score(model_pr, X, y, scoring='r2', cv=cv, n_jobs=-1))
  fit['Gamma'] = mean(cross_val_score(model_gr, X, y, scoring='r2', cv=cv, n_jobs=-1))
  fit['Inverse'] = mean(cross_val_score(model_igr, X, y, scoring='r2', cv=cv, n_jobs=-1))
  fit['SupportVM'] = mean(cross_val_score(model_svm, X, y, scoring='r2', cv=cv, n_jobs=-1))
  fit['Linear SVM'] = mean(cross_val_score(model_lsvm, X, y, scoring='r2', cv=cv, n_jobs=-1))
  fit['NuSupportVM'] = mean(cross_val_score(model_nusvm, X, y, scoring='r2', cv=cv, n_jobs=-1))
  fit['KNNeighbors'] = mean(cross_val_score(model_knnr, X, y, scoring='r2', cv=cv, n_jobs=-1))
  fit['KNNeighborsD'] = mean(cross_val_score(model_knnrd, X, y, scoring='r2', cv=cv, n_jobs=-1))
  fit['GaussianP'] = mean(cross_val_score(model_gpr, X, y, scoring='r2', cv=cv, n_jobs=-1))
  fit['Dec Forest'] = mean(cross_val_score(model_df, X, y, scoring='r2', cv=cv, n_jobs=-1))
  fit['Extra Trees'] = mean(cross_val_score(model_etr, X, y, scoring='r2', cv=cv, n_jobs=-1))
  fit['AdaBoost DT'] = mean(cross_val_score(model_abr, X, y, scoring='r2', cv=cv, n_jobs=-1))
  fit['Grad. Boost'] = mean(cross_val_score(model_gbr, X, y, scoring='r2', cv=cv, n_jobs=-1))
  fit['HG Boost'] = mean(cross_val_score(model_hgbr, X, y, scoring='r2', cv=cv, n_jobs=-1))
  fit['Voting'] = mean(cross_val_score(model_vr, X, y, scoring='r2', cv=cv, n_jobs=-1))
  fit['Stacking'] = mean(cross_val_score(model_sr, X, y, scoring='r2', cv=cv, n_jobs=-1))
  fit['XGBoost'] = mean(cross_val_score(model_xgb, X, y, scoring='r2', cv=cv, n_jobs=-1))
  fit['NeuralNet'] = mean(cross_val_score(model_nn, X, y, scoring='r2', cv=cv, n_jobs=-1))

  # Add the model to another dict; make sure the keys have the same names as the list above
  models['OLS'] = model_ols
  models['Ridge'] = model_rr
  models['Lasso'] = model_lr
  models['LARS'] = model_llr
  models['Bayesian'] = model_br
  models['Poisson'] = model_pr
  models['Gamma'] = model_gr
  models['Inverse'] = model_igr
  models['SupportVM'] = model_svm
  models['Linear SVM'] = model_lsvm
  models['NuSupportVM'] = model_nusvm
  models['KNNeighbors'] = model_knnr
  models['KNNeighborsD'] = model_knnrd
  models['GaussianP'] = model_gpr
  models['Dec Forest'] = model_df
  models['Extra Trees'] = model_etr
  models['AdaBoost DT'] = model_abr
  models['Grad. Boost'] = model_gbr
  models['HG Boost'] = model_hgbr
  models['Voting'] = model_vr
  models['Stacking'] = model_sr
  models['XGBoost'] = model_xgb
  models['NeuralNet'] = model_nn

  # Add the fit dictionary to a new DataFrame, sort, extract the top row, use it to retrieve the model object from the models dictionary
  df_fit = pd.DataFrame({'R-squared':fit})
  df_fit.sort_values(by=['R-squared'], ascending=False, inplace=True)
  best_model = df_fit.index[0]
  print(df_fit)

  return models[best_model].fit(X, y)