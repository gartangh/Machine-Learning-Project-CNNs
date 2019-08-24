import numpy as np
import csv
import time

file = 'Xception/Xception_features_test_20181120_000206'

features_test = np.load(open(file, 'rb'))
label_strings = ['bobcat', 'chihuahua', 'collie', 'dalmatian', 'german_shepherd', 'leopard', 'lion', 'persian_cat',
                 'siamese_cat', 'tiger', 'wolf']


def write_predictions_to_csv(predictions, out_path, label_strings):
	"""Writes the predictions to a csv file.
	Assumes the predictions are ordered by test interval id."""
	with open(out_path, 'w') as outfile:
		# Initialise the writer
		csvwriter = csv.writer(
			outfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		# Write the header
		row_to_write = ['Id'] + [label for label in label_strings]
		csvwriter.writerow(row_to_write)
		# Write the rows using 18 digit precision
		for idx, prediction in enumerate(predictions):
			assert len(prediction) == len(label_strings)
			csvwriter.writerow([str(idx + 1)] + ["%.18f" % p for p in prediction])


def generate_unique_filename(basename, file_ext):
	"""Adds a timestamp to filenames for easier tracking of submissions, models, etc."""
	timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
	return basename + '_' + timestamp + '.' + file_ext


unique_file_name = generate_unique_filename('pred', 'csv')
write_predictions_to_csv(features_test, f'Predictions/{unique_file_name}', label_strings)
