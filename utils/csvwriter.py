import csv

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

    def writerows(self, filename, centers_i, centers_j, embeddings):
        for i, j, em in zip(centers_i, centers_j, embeddings):
            em = [f'{val:.{self.precision}e}' for val in em]
            self.writer.writerow([filename, i, j, em])

    def close(self):
        self.writer = None
        self.fid.close()