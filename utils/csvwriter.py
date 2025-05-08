import csv
import numbers

class CSVWriter:
    """
    Wrapper for creating and writing to a csv or tsv file.
    """
    def __init__(self, filename: str, precision: int = 3):
        self.fid = open(filename, 'w')
        self.precision = precision
        delimiter = '\t' if filename.endswith('.tsv') else ','
        self.writer = csv.writer(self.fid, delimiter=delimiter)

    def write_header(self, header):
        self.writer.writerow(header)

    def format(self, val):
        if hasattr(val, '__iter__'): # list, tuple, ndarray
            return [self.format(v) for v in val]
        if isinstance(val, numbers.Real) and not isinstance(val, numbers.Integral):
            # floating type, including numpy types
            return f'{val:.{self.precision}e}'
        return val

    def writerows(self, filename, columns):
        for row in zip(*columns):
            row = list(row)
            for idx in range(len(row)):
                row[idx] = self.format(row[idx])
            self.writer.writerow([filename] + row)

    def writerow(self, filename, columns):
        columns = list(columns)
        for idx in range(len(columns)):
            columns[idx] = self.format(columns[idx])
        self.writer.writerow([filename] + columns)

    def close(self):
        self.writer = None
        self.fid.close()