# this will serve as the custom-made ORM for this project. It will inherit the PostgreSQLConnector class
# and build on it. This is for re-usability (I.E. can copy over PostgreSQLConnector to any project, without worrying
# about project specific ORM code).

from config.configs import db_config
from database.database import PostgreSQLConnector

import logging


class BagelsORM(PostgreSQLConnector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def login(self, user):


orm = BagelsORM(**db_config)
