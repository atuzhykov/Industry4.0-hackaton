def data_fixer(df):
    valid_data = []
    for record in df.values:
        if 'None' not in record:
            valid_data.append(record)
    valid_data = [list(map(float, sublist)) for sublist in valid_data]
    return valid_data