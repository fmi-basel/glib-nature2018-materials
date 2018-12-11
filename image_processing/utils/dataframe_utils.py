def filter_by(dataframe, key, positive_list):
    '''removes all rows whose key value is not in positive_list.

    :param dataframe: dataframe to be filtered.
    :param key: column name to be used for filtering.
    :param positive_list: values to be kept.

    :return filtered_dataframe: filtered dataframe.

    '''
    positive_list = list(set(positive_list))  # make unique.
    return dataframe.loc[dataframe[key].isin(positive_list)]
