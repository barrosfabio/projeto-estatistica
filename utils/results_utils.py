import csv

"""
    This method writes a data_frame to CSV
"""
def write_csv(file_name, data_frame):
    csv_file_path = file_name + '.csv'
    header = list(data_frame.columns.values)

    print('Saving file to path: {}'.format(csv_file_path))

    with open(csv_file_path, 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL, dialect='excel')

        # Write the header of the table
        filewriter.writerow(header)

        # Write all the other rows
        for index, row in data_frame.iterrows():
            filewriter.writerow(row)