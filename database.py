import csv
import logging
from io import StringIO
from typing import Dict, List, Optional

import pandas as pd
import psycopg2
from psycopg2 import sql
from sqlalchemy import create_engine


class PostgresDB:
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

    def run_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Runs a SQL query on the database.

        Args:
            query (str): The SQL query to execute.
            params (Optional[Dict]): Optional parameters for the SQL query.

        Returns:
            pd.DataFrame: A DataFrame with the results of the query.
        """
        with self.conn.cursor() as cur:
            try:
                cur.execute(query, params)
                if query.lower().startswith("select"):
                    data = cur.fetchall()
                    column_names = [desc[0] for desc in cur.description]
                    df = pd.DataFrame(data, columns=column_names)
                    self.logger.info(f'Returned dataframe of shape: {df.shape[0]} x {df.shape[1]}')
                    return df
                else:
                    self.conn.commit()
                    self.logger.info("Query executed successfully")
            except Exception as e:
                self.logger.error(f"Error executing query: {e}")

    def create_table(self, table_name: str, columns: Dict[str, str]):
        """
        Create a new table in the database.

        Args:
            table_name (str): The name of the table to create.
            columns (Dict[str, str]): A dictionary with column names and their data types.
        """
        with self.conn.cursor() as cur:
            try:
                columns_sql = ", ".join([f"{col} {data_type}" for col, data_type in columns.items()])
                create_table_query = sql.SQL("CREATE TABLE {} ({});").format(
                    sql.Identifier(table_name),
                    sql.SQL(columns_sql)
                )
                cur.execute(create_table_query)
                self.conn.commit()
                self.logger.info(f"Table '{table_name}' created successfully")
            except Exception as e:
                self.logger.error(f"Error creating table: {e}")

    def add_primary_key(self, table_name: str, column: str, constraint_name: Optional[str] = None):
        """
        Add primary key to the table.

        Args:
            table_name (str): The name of the table to modify.
            column (str): The column to set as primary key.
            constraint_name (Optional[str]): The name of the constraint. Defaults to "{table_name}_pkey".
        """
        with self.conn.cursor() as cur:
            try:
                if constraint_name is None:
                    constraint_name = f"{table_name}_pkey"
                query = sql.SQL("ALTER TABLE {} ADD CONSTRAINT {} PRIMARY KEY ({});").format(
                    sql.Identifier(table_name),
                    sql.Identifier(constraint_name),
                    sql.Identifier(column)
                )
                cur.execute(query)
                self.conn.commit()
                self.logger.info(f"Primary key added to '{table_name}' successfully")
            except Exception as e:
                self.logger.error(f"Error adding primary key to table: {e}")

    # Continue in a similar manner for other methods...

    def modify_table(self, table_name, schema_name, action, **kwargs):
        with self.conn.cursor() as cur:
            try:
                if action == "add_primary_key":
                    column = kwargs.get("column")
                    constraint_name = kwargs.get("constraint_name", f"{table_name}_pkey")
                    query = sql.SQL("ALTER TABLE {}.{} ADD CONSTRAINT {} PRIMARY KEY ({});").format(
                        sql.Identifier(schema_name),
                        sql.Identifier(table_name),
                        sql.Identifier(constraint_name),
                        sql.Identifier(column)
                    )
                elif action == "add_unique_key":
                    columns = kwargs.get("columns")
                    constraint_name = kwargs.get("constraint_name", f"{table_name}_unique")
                    query = sql.SQL("ALTER TABLE {}.{} ADD CONSTRAINT {} UNIQUE ({});").format(
                        sql.Identifier(schema_name),
                        sql.Identifier(table_name),
                        sql.Identifier(constraint_name),
                        sql.SQL(", ").join([sql.Identifier(col) for col in columns])
                    )
                elif action == "add_foreign_key":
                    column = kwargs.get("column")
                    reference_table = kwargs.get("reference_table")
                    reference_column = kwargs.get("reference_column")
                    constraint_name = kwargs.get("constraint_name", f"{table_name}_fk")
                    query = sql.SQL("ALTER TABLE {} ADD CONSTRAINT {} FOREIGN KEY ({}) REFERENCES {} ({});").format(
                        sql.Identifier(table_name),
                        sql.Identifier(constraint_name),
                        sql.Identifier(column),
                        sql.Identifier(reference_table),
                        sql.Identifier(reference_column)
                    )
                elif action == "add_sequence":
                    column = kwargs.get("column")
                    sequence_name = kwargs.get("sequence_name", f"{table_name}_{column}_seq")
                    query = sql.SQL("ALTER TABLE {} ALTER COLUMN {} SET DEFAULT nextval('{}');").format(
                        sql.Identifier(table_name),
                        sql.Identifier(column),
                        sql.Identifier(sequence_name)
                    )
                else:
                    raise ValueError("Invalid action")

                cur.execute(query)
                self.conn.commit()
                print(f"Table '{table_name}' modified successfully")
            except Exception as e:
                print("Error modifying table: ", e)

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


db_connector = PostgresDB(host='172.21.64.1',
                          user='elijah',
                          dbname='market_data',
                          port='5432',
                          password='Poodle!3')
