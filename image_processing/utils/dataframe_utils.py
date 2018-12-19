def filter_by(dataframe, key, positive):
    '''removes all rows whose key value is not in positive_list.

    :param dataframe: dataframe to be filtered.
    :param key: column name to be used for filtering.
    :param positive: list of values to be kept or function
                     returning True for values to be kept.

    :return filtered_dataframe: filtered dataframe.

    '''
    if callable(positive):
        positive_list = [x for x in dataframe[key].unique() if positive(x)]
    else:
        positive_list = list(set(positive))  # make unique.
    return dataframe.loc[dataframe[key].isin(positive_list)]
