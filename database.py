import csv
import logging
from io import StringIO
from typing import Any
from typing import Dict, List, Optional
from typing import Tuple
from typing import Union

import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.sql import Composed
from sqlalchemy import create_engine
from sqlalchemy import inspect


class PostgreSQLConnector:
    """
    A class used to manage a PostgreSQL database.

    ...

    Attributes
    ----------
    host : str
        the host address of the PostgreSQL database
    port : str
        the port number of the PostgreSQL database
    user : str
        the user name to access the PostgreSQL database
    password : str
        the password to access the PostgreSQL database
    dbname : str
        the name of the PostgreSQL database
    conn : psycopg2.extensions.connection
        the connection object to the PostgreSQL database
    logger : logging.Logger
        the logger object to log events

    Methods
    -------
    connect():
        Connects to the PostgreSQL database.
    disconnect():
        Disconnects from the PostgreSQL database.
    run_query(query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        Runs a SQL query on the database.
    create_table(table_name: str, columns: Dict[str, str]):
        Creates a new table in the database.
    add_primary_key(table_name: str, column: str, constraint_name: Optional[str] = None):
        Adds a primary key to a table in the database.
    add_unique_key(table_name: str, columns: List[str], constraint_name: Optional[str] = None):
        Adds a unique key to a table in the database.
    add_foreign_key(table_name: str, column: str, reference_table: str, reference_column: str, constraint_name: Optional[str] = None):
        Adds a foreign key to a table in the database.
    add_sequence(table_name: str, column: str, sequence_name: Optional[str] = None):
        Adds a sequence to a table in the database.
    create_engine() -> create_engine:
        Creates a SQLAlchemy engine connected to the database.
    insert_dataframe(dataframe: pd.DataFrame, table_name: str, if_exists: str = 'append', index: bool = False):
        Inserts a pandas DataFrame into the database.
    psql_insert_copy(table, conn, keys, data_iter):
        Helper function to use PostgreSQL's COPY command for faster inserts.
    """

    def __init__(self, host: str, port: str, user: str, password: str, dbname: str):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.dbname = dbname
        self.conn = None

        self.logger = logging.getLogger(__name__)

    def connect(self):
        """Connects to the PostgreSQL database."""
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                dbname=self.dbname
            )
            self.logger.info("Connection successful")
        except Exception as e:
            self.logger.error(f"Error connecting to the database: {e}")

    def disconnect(self):
        """Disconnects from the PostgreSQL database."""
        if self.conn is not None:
            self.conn.close()
            self.logger.info("Connection closed")

    def run_query(
            self, query: Union[str, Composed], params: Optional[Tuple] = None, return_df: bool = True,
            fetch_one: bool = False
    ) -> Optional[Union[pd.DataFrame, Any]]:
        """
        Execute a query on the database.

        Args:
            query (Union[str, Composed]): SQL query as a string or psycopg2 Composed object.
            params (Optional[Tuple], optional): Tuple of parameters to use in the query. Defaults to None.
            return_df (bool, optional): Whether to return the results as a pandas DataFrame.
                                        Defaults to True. If False, None is returned.
            fetch_one (bool, optional): Whether to return a single value.
                                        Defaults to False. If True, returns a single value from the query result.

        Returns:
            Optional[Union[pd.DataFrame, Any]]: Result of the query as a pandas DataFrame, if return_df is True and
            the query
            retrieves data. Otherwise, None is returned. If fetch_one is True, returns a single value from the query
            result.
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, params)
                if fetch_one:
                    data = cur.fetchone()
                    self.conn.commit()
                    return data[0] if data else None
                elif return_df:
                    data = cur.fetchall()
                    colnames = [desc[0] for desc in cur.description]
                    return pd.DataFrame(data, columns=colnames)
                else:
                    self.conn.commit()
                    return None
        except Exception as e:
            self.logger.error(f'Error occurred while executing query: {e}')
            raise e

    def create_table(self, table_name: str, columns: Dict[str, str], schema: str = 'public') -> None:
        """
        Create a table in a specified schema.

        Args:
            table_name (str): Table name.
            columns (Dict[str, str]): Dictionary with column names as keys and data types as values.
            schema (str, optional): The schema in which to create the table. Defaults to 'public'.

        Examples:
            connector.create_table('users', {'id': 'SERIAL', 'name': 'VARCHAR(100)', 'email': 'VARCHAR(100)'},
            'myschema')
        """
        try:
            query = sql.SQL('CREATE TABLE IF NOT EXISTS {}.{} ({})').format(
                sql.Identifier(schema),
                sql.Identifier(table_name),
                sql.SQL(', ').join(
                    sql.SQL('{} {}').format(
                        sql.Identifier(column), sql.SQL(data_type)
                    ) for column, data_type in columns.items()
                )
            )
            self.run_query(query, return_df=False)
            self.logger.info(f'Table {table_name} created successfully in schema {schema}.')
        except Exception as e:
            self.logger.error(f'Error occurred while creating table: {e}')
            raise e

    def table_exists(self, table_name: str, schema: str = 'public') -> bool:
        """
        Checks if a table exists in a specified schema.

        Args:
            table_name (str): Table name.
            schema (str, optional): The schema in which the table resides. Defaults to 'public'.

        Returns:
            bool: True if the table exists, False otherwise.
        """
        engine = self.create_engine()
        inspector = inspect(engine)
        return table_name in inspector.get_table_names(schema=schema)

    def add_primary_key(self, table_name: str, column_name: str, schema: str = 'public') -> None:
        """
        Add a primary key to a table.

        Args:
            table_name (str): Table name.
            column_name (str): Column name.
            schema (str, optional): The schema in which the table resides. Defaults to 'public'.

        Examples:
            >> connector.add_primary_key('users', 'id', 'myschema')
        """
        query = sql.SQL('ALTER TABLE {schema}.{table} ADD PRIMARY KEY ({column})').format(
            schema=sql.Identifier(schema),
            table=sql.Identifier(table_name),
            column=sql.Identifier(column_name)
        )
        self.run_query(query, return_df=False)

    def add_foreign_key(self, table_name: str, column_name: str, ref_table: str, ref_column: str,
                        schema: str = 'public') -> None:
        """
        Add a foreign key to a table.

        Args:
            table_name (str): Table name.
            column_name (str): Column name.
            ref_table (str): Referenced table name.
            ref_column (str): Referenced column name.
            schema (str, optional): The schema in which the table resides. Defaults to 'public'.

        Examples:
            >> connector.add_foreign_key('orders', 'user_id', 'users', 'id', 'myschema')
        """
        query = sql.SQL(
            'ALTER TABLE {schema}.{table} ADD FOREIGN KEY ({column}) REFERENCES {schema}.{ref_table} ({ref_column})').format(
            schema=sql.Identifier(schema),
            table=sql.Identifier(table_name),
            column=sql.Identifier(column_name),
            ref_table=sql.Identifier(ref_table),
            ref_column=sql.Identifier(ref_column)
        )
        self.run_query(query, return_df=False)

    def add_unique_key(self, table_name: str, columns: List[str], constraint_name: str, schema: str = 'public') -> None:
        """
        Add a unique constraint to a table.

        Args:
            table_name (str): Table name.
            columns (List[str]): List of column names that compose the unique key.
            constraint_name (str): Name of the unique constraint.
            schema (str, optional): The schema in which the table resides. Defaults to 'public'.

        Example:
            >> connector.add_unique_key('models', ['date_trained', 'features', 'symbol', 'days_forecast'],
            'models_unique_key', 'myschema')

        Raises:
            Exception: If an error occurs while adding the unique key.
        """
        try:
            query = sql.SQL(
                'ALTER TABLE {}.{} ADD CONSTRAINT {} UNIQUE ({})'
            ).format(
                sql.Identifier(schema),
                sql.Identifier(table_name),
                sql.Identifier(constraint_name),
                sql.SQL(', ').join(map(sql.Identifier, columns))
            )
            self.run_query(query, return_df=False)
            self.logger.info(
                f'Unique key {constraint_name} added successfully to table {table_name} in schema {schema}.'
            )
        except Exception as e:
            self.logger.error(f'Error occurred while adding unique key: {e}')
            raise e

    def create_engine(self):
        return create_engine(f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}")

    def insert_dataframe(self, dataframe, **kwargs):
        try:
            engine = self.create_engine()
            dataframe.to_sql(**kwargs, con=engine, method=self.psql_insert_copy)
            print(f"Dataframe inserted into db successfully.")
        except Exception as e:
            print("Error inserting dataframe: ", e)

    @staticmethod
    def psql_insert_copy(table, conn, keys, data_iter):
        # gets a DBAPI connection that can provide a cursor
        dbapi_conn = conn.connection
        with dbapi_conn.cursor() as cur:
            s_buf = StringIO()
            writer = csv.writer(s_buf)
            writer.writerows(data_iter)
            s_buf.seek(0)

            columns = ', '.join('"{}"'.format(k) for k in keys)
            if table.schema:
                table_name = '{}.{}'.format(table.schema, table.name)
            else:
                table_name = table.name

            sql = 'COPY {} ({}) FROM STDIN WITH CSV'.format(
                table_name, columns)
            cur.copy_expert(sql=sql, file=s_buf)

    def insert_and_return_id(
            self, table_name: str, columns: Dict[str, Any], schema: str = 'public'
    ) -> int:
        """
        Insert a row into a specified table and return the generated id.

        Args:
            table_name (str): Table name.
            columns (Dict[str, Any]): Dictionary with column names as keys and data to be inserted as values.
            schema (str, optional): The schema in which the table resides. Defaults to 'public'.

        Returns:
            int: The id of the row that was inserted.

        Examples:
            >> connector.insert_and_return_id('users', {'name': 'John', 'email': 'john@example.com'}, 'myschema')
        """
        query = sql.SQL(
            'INSERT INTO {}.{} ({}) VALUES ({}) RETURNING id'
        ).format(
            sql.Identifier(schema),
            sql.Identifier(table_name),
            sql.SQL(', ').join(map(sql.Identifier, columns.keys())),
            sql.SQL(', ').join(sql.Placeholder() * len(columns))
        )
        params = tuple(columns.values())
        print(query)
        print(params)
        return self.run_query(query, params=params, return_df=False, fetch_one=True)



db_connector = PostgreSQLConnector(
    host='172.18.240.1',
    user='elijah',
    dbname='market_data',
    port='5432',
    password='Poodle!3'
)