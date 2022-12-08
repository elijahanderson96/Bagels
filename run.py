from datetime import date
from traceback import print_exc

from config.common import *
from data.iex import Pipeline
from data.labels import Prices
from models import models

today = date.today()
# Month abbreviation, day and year
today = today.strftime("%b_%d_%Y")

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    DataGetter = Pipeline()
    DataGetter.update_data()
    prices = Prices()
    prices.update_db()
    for company_size, symbols in sym_mkcap_mappings.items():
        symbols = [symbol for symbol in symbols if symbol not in EXCLUDE_LIST]
        if company_size == 'SMALL_CAP':
            try:
                model = models.SectorModel(sector=symbols, model_type='fundamental_valuations')
                model.train(validate=True, interpolate_data=False, interpolate_labels=False)
                model.predict()
                model.save()
                val_loss = min(model.history.history['val_loss'])
                index = model.history.history['val_loss'].index(val_loss)
                training_loss = model.history.history['loss'][index]
                model.all_scores['val_loss'] = val_loss
                model.all_scores['loss'] = training_loss
                model.all_scores.to_sql(f'predictions_{company_size}_{today}', con=POSTGRES_URL, schema='model_predictions',
                                        if_exists='replace',
                                        index=False)
            except:
                print(print_exc())
                print(f'Likely missing test data or labels')
