from data.iex import Pipeline
import logging
logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    DataGetter = Pipeline()
    DataGetter.run(['C', 'BAC', 'JPM'])
