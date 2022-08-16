from data.iex import Pipeline
from data.labels import Prices
from data.transforms import interpolate

if __name__ == '__main__':
    DataGetter = Pipeline()
    DataGetter.run(['A', 'AA', 'AAC', 'AACG', 'AACI'])
