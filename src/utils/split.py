from sklearn.model_selection import train_test_split


def random_splits(dataset, split_portions):
    total_portion = sum(split_portions)

    # Check that the split portions are global by sampling the total split portion
    if total_portion > 1.0:
        raise ValueError('The splits must not add up to more than 1.0!')
    elif total_portion != 1.0:
        _, dataset = train_test_split(dataset, test_size=total_portion)
        split_portions = [portion / total_portion for portion in split_portions]

    # Take the splits tracking the global % taken so far
    splits = []
    current_fraction = 1.0
    for portion in split_portions[:-1]:
        scaled_portion = portion / current_fraction
        split, dataset = train_test_split(dataset, test_size=scaled_portion)
        splits.append(split)
        current_fraction -= portion

    # Include the last split
    splits.append(dataset)
    return splits


def sorted_splits(dataset, sorting_key, split_portions, ascending=True, inplace=False):
    total_elements = len(dataset)
    sorted_dataset = dataset.sort_values(sorting_key, ascending=ascending, inplace=False)

    # Compute splits by slicing over the sorted dataframe
    splits = []
    range_start = 0
    portion_sum = 0.0
    for portion in split_portions:
    	portion_end = portion + portion_sum
    	range_end = int(round(portion_end * total_elements))

    	split = sorted_dataset[range_start:range_end]
    	range_start = range_end
    	portion_sum = portion_end
    	splits.append(split)
    return splits


def column_splits(dataset, column='split'):
	values = dataset[column].unique()
	splits = []
	for value in values:
		value_split = dataset[dataset[column] == value]
		splits.append(value_split)
	return splits

