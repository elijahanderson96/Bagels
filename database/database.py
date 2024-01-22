import csv
import logging
from io import StringIO
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.sql import Composed
from sqlalchemy import create_engine, inspect

from config.configs import db_config


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

    def __init__(
        self, host: str, port: str, user: str, password: str, dbname: str = None
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.dbname = dbname

        self.logger = logging.getLogger(__name__)

    def connect(self):
        """Connects to the PostgreSQL database."""
        try:
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                dbname=self.dbname if self.dbname else None,
            )
            return conn
        except Exception as e:
            self.logger.error(f"Error connecting to the database: {e}")

    def insert_row(self, table, data, schema=None):
        """
        Insert a row into a table.

        Parameters:
        table (str): The name of the table.
        data (dict): A dictionary of column-value pairs to insert.
        schema (str, optional): The schema name. Defaults to None.
        """
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["%s" for _ in data])
        query = sql.SQL(
            "INSERT INTO {schema}.{table} ({columns}) VALUES ({placeholders})"
        ).format(
            schema=sql.Identifier(schema) if schema else sql.SQL("public"),
            table=sql.Identifier(table),
            columns=sql.SQL(columns),
            placeholders=sql.SQL(placeholders),
        )
        try:
            conn = self.connect()
            with conn.cursor() as cursor:
                cursor.execute(query, list(data.values()))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error inserting row into {table}: {e}")
            raise

    def run_query(
        self,
        query: Union[str, Composed],
        params: Optional[Dict] = None,
        return_df: bool = True,
        fetch_one: bool = False,
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
            conn = self.connect()
            with conn.cursor() as cur:
                cur.execute(query, params)
                conn.commit()  # Commit changes here

                if fetch_one:
                    data = cur.fetchone()
                    conn.commit()
                    return data[0] if data else None
                elif return_df:
                    data = cur.fetchall()
                    colnames = [desc[0] for desc in cur.description]
                    return pd.DataFrame(data, columns=colnames)
                else:
                    conn.commit()
                    return None
        except Exception as e:
            self.logger.error(f"Error occurred while executing query: {e}")
            raise e

    def create_database(self, dbname: str):
        """
        Create a new database.

        Args:
            dbname (str): The name of the database to create.
        """
        # Establish a new connection to the PostgreSQL server
        conn = psycopg2.connect(
            dbname="postgres", user=self.user, host=self.host, password=self.password
        )
        conn.autocommit = True  # Enable autocommit mode for this transaction

        with conn.cursor() as cur:
            try:
                cur.execute(f"CREATE DATABASE {dbname};")
                self.logger.info(f"Database {dbname} created successfully.")
            except psycopg2.Error as e:
                if "already exists" in str(e):
                    self.logger.info(f"Database {dbname} already exists.")
                else:
                    self.logger.error(f"Error occurred while creating database: {e}")
                    raise e
        conn.close()  # Close the connection

    def create_schema(self, schema_name: str) -> None:
        """
        Create a new schema in the database.

        Args:
            schema_name (str): The name of the schema to be created.

        Examples:
            >> connector.create_schema('myschema')
        """
        query = sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
            sql.Identifier(schema_name)
        )
        self.run_query(query, return_df=False)
        self.logger.info(f"Schema {schema_name} created successfully.")

    def create_enum_type(
        self, enum_name: str, values: List[str], schema: str = "public"
    ) -> None:
        """
        Create an ENUM type in the specified schema.

        Args:
            enum_name (str): The name of the ENUM type.
            values (List[str]): The allowable values for the ENUM type.
            schema (str, optional): The schema in which to create the ENUM type. Defaults to 'public'.
        """
        try:
            query = sql.SQL(
                "DO $$ BEGIN CREATE TYPE {}.{} AS ENUM ({}); EXCEPTION WHEN duplicate_object THEN null; END $$;"
            ).format(
                sql.Identifier(schema),
                sql.Identifier(enum_name),
                sql.SQL(", ").join(sql.Literal(value) for value in values),
            )
            self.run_query(query, return_df=False)
            self.logger.info(
                f"ENUM type {enum_name} created successfully in schema {schema}."
            )
        except Exception as e:
            self.logger.error(f"Error occurred while creating ENUM type: {e}")
            raise e

    def create_table(
        self, table_name: str, columns: Dict[str, str], schema: str = "dpc"
    ) -> None:
        """
        Create a table in a specified schema.

        Args:
            table_name (str): Table name.
            columns (Dict[str, str]): Dictionary with column names as keys and data types as values.
            schema (str, optional): The schema in which to create the table. Defaults to 'dpc'.

        Examples:
            connector.create_table('users', {'id': 'SERIAL', 'name': 'VARCHAR(100)', 'email': 'VARCHAR(100)'},
            'myschema')
        """
        try:
            query = sql.SQL("CREATE TABLE IF NOT EXISTS {}.{} ({})").format(
                sql.Identifier(schema),
                sql.Identifier(table_name),
                sql.SQL(", ").join(
                    sql.SQL("{} {}").format(sql.Identifier(column), sql.SQL(data_type))
                    for column, data_type in columns.items()
                ),
            )
            self.run_query(query, return_df=False)
            self.logger.info(
                f"Table {table_name} created successfully in schema {schema}."
            )
        except Exception as e:
            self.logger.error(f"Error occurred while creating table: {e}")
            raise e

    def drop_table_if_exists(self, table_name: str, schema: str = "dpc") -> None:
        """
        Drop a table if it exists.

        Args:
            table_name (str): Table name.
            schema (str, optional): The schema in which the table resides. Defaults to 'dpc'.
        """
        try:
            query = sql.SQL("DROP TABLE IF EXISTS {}.{} CASCADE").format(
                sql.Identifier(schema),
                sql.Identifier(table_name),
            )
            self.run_query(query, return_df=False)
            self.logger.info(
                f"Table {table_name} dropped successfully from schema {schema}."
            )
        except Exception as e:
            self.logger.error(f"Error occurred while dropping table: {e}")
            raise e

    def table_exists(self, table_name: str, schema: str = "dpc") -> bool:
        """
        Checks if a table exists in a specified schema.

        Args:
            table_name (str): Table name.
            schema (str, optional): The schema in which the table resides. Defaults to 'dpc'.

        Returns:
            bool: True if the table exists, False otherwise.
        """
        engine = self.create_engine()
        inspector = inspect(engine)
        return table_name in inspector.get_table_names(schema=schema)

    def add_primary_key(
        self, table_name: str, column_name: str, schema: str = "dpc"
    ) -> None:
        """
        Add a primary key to a table.

        Args:
            table_name (str): Table name.
            column_name (str): Column name.
            schema (str, optional): The schema in which the table resides. Defaults to 'dpc'.

        Examples:
            >> connector.add_primary_key('users', 'id', 'myschema')
        """
        query = sql.SQL(
            "ALTER TABLE {schema}.{table} ADD PRIMARY KEY ({column})"
        ).format(
            schema=sql.Identifier(schema),
            table=sql.Identifier(table_name),
            column=sql.Identifier(column_name),
        )
        self.run_query(query, return_df=False)

    def drop_constraint_if_exists(
        self, table_name: str, constraint_name: str, schema: str = "dpc"
    ) -> None:
        """
        Drop a constraint from a table if it exists.

        Args:
            table_name (str): Table name.
            constraint_name (str): Constraint name.
            schema (str, optional): The schema in which the table resides. Defaults to 'dpc'.

        Examples:
            >> connector.drop_constraint_if_exists('users', 'user_email_key', 'myschema')
        """
        query = sql.SQL(
            "ALTER TABLE {schema}.{table} DROP CONSTRAINT IF EXISTS {constraint};"
        ).format(
            schema=sql.Identifier(schema),
            table=sql.Identifier(table_name),
            constraint=sql.Identifier(constraint_name),
        )
        self.run_query(query, return_df=False)

    def add_foreign_key(
        self,
        table_name: str,
        column_name: str,
        ref_table: str,
        ref_column: str,
        schema: str = "dpc",
    ) -> None:
        """
        Add a foreign key to a table.

        Args:
            table_name (str): Table name.
            column_name (str): Column name.
            ref_table (str): Referenced table name.
            ref_column (str): Referenced column name.
            schema (str, optional): The schema in which the table resides. Defaults to 'dpc'.

        Examples:
            >> connector.add_foreign_key('orders', 'user_id', 'users', 'id', 'myschema')
        """
        query = sql.SQL(
            "ALTER TABLE {schema}.{table} ADD FOREIGN KEY ({column}) REFERENCES {schema}.{ref_table} ({ref_column})"
        ).format(
            schema=sql.Identifier(schema),
            table=sql.Identifier(table_name),
            column=sql.Identifier(column_name),
            ref_table=sql.Identifier(ref_table),
            ref_column=sql.Identifier(ref_column),
        )
        self.run_query(query, return_df=False)

    def add_unique_key(
        self,
        table_name: str,
        columns: List[str],
        constraint_name: str,
        schema: str = "dpc",
    ) -> None:
        """
        Add a unique constraint to a table.

        Args:
            table_name (str): Table name.
            columns (List[str]): List of column names that compose the unique key.
            constraint_name (str): Name of the unique constraint.
            schema (str, optional): The schema in which the table resides. Defaults to 'dpc'.

        Example:
            >> connector.add_unique_key('models', ['date_trained', 'features', 'symbol', 'days_forecast'],
            'models_unique_key', 'myschema')

        Raises:
            Exception: If an error occurs while adding the unique key.
        """
        try:
            query = sql.SQL("ALTER TABLE {}.{} ADD CONSTRAINT {} UNIQUE ({})").format(
                sql.Identifier(schema),
                sql.Identifier(table_name),
                sql.Identifier(constraint_name),
                sql.SQL(", ").join(map(sql.Identifier, columns)),
            )
            self.run_query(query, return_df=False)
            self.logger.info(
                f"Unique key {constraint_name} added successfully to table {table_name} in schema {schema}."
            )
        except Exception as e:
            self.logger.error(f"Error occurred while adding unique key: {e}")
            raise e

    def create_engine(self):
        return create_engine(
            f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}"
        )

    def insert_dataframe(self, dataframe, **kwargs):
        try:
            engine = create_engine(
                f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}"
            )
            dataframe.to_sql(**kwargs, con=engine, method=self.psql_insert_copy)
            engine.dispose()  # close the engine
            self.logger.info(f"Dataframe inserted into db successfully.")
        except Exception as e:
            self.logger.error("Error inserting dataframe: ", e)
            raise e

    @staticmethod
    def psql_insert_copy(table, conn, keys, data_iter):
        # gets a DBAPI connection that can provide a cursor
        dbapi_conn = conn.connection
        with dbapi_conn.cursor() as cur:
            s_buf = StringIO()
            writer = csv.writer(s_buf)
            writer.writerows(data_iter)
            s_buf.seek(0)

            columns = ", ".join('"{}"'.format(k) for k in keys)
            if table.schema:
                table_name = "{}.{}".format(table.schema, table.name)
            else:
                table_name = table.name

            sql = "COPY {} ({}) FROM STDIN WITH CSV".format(table_name, columns)
            cur.copy_expert(sql=sql, file=s_buf)

    def insert_and_return_id(
        self, table_name: str, columns: Dict[str, Any], schema: str = "dpc"
    ) -> int:
        """
        Insert a row into a specified table and return the generated id.

        Args:
            table_name (str): Table name.
            columns (Dict[str, Any]): Dictionary with column names as keys and data to be inserted as values.
            schema (str, optional): The schema in which the table resides. Defaults to 'dpc'.

        Returns:
            int: The id of the row that was inserted.

        Examples:
            >> connector.insert_and_return_id('users', {'name': 'John', 'email': 'john@example.com'}, 'myschema')
        """
        query = sql.SQL("INSERT INTO {}.{} ({}) VALUES ({}) RETURNING id").format(
            sql.Identifier(schema),
            sql.Identifier(table_name),
            sql.SQL(", ").join(map(sql.Identifier, columns.keys())),
            sql.SQL(", ").join(sql.Placeholder() * len(columns)),
        )
        params = tuple(columns.values())
        return self.run_query(query, params=params, return_df=False, fetch_one=True)

    def create_sequence(
        self, table_name: str, column_name: str, schema: str = "dpc"
    ) -> None:
        """
        Create a sequence and set it as the default value for a table's column.

        Args:
            table_name (str): The name of the table.
            column_name (str): The name of the column.
            schema (str, optional): The schema in which the table resides. Defaults to 'dpc'.
        """
        try:
            sequence_name = f"{table_name}_{column_name}_seq"

            create_sequence_query = sql.SQL(
                "CREATE SEQUENCE IF NOT EXISTS {sequence}"
            ).format(sequence=sql.Identifier(sequence_name))

            self.run_query(create_sequence_query, return_df=False)

            alter_table_query = sql.SQL(
                "ALTER TABLE {schema}.{table} ALTER COLUMN {column} SET DEFAULT nextval('{sequence}')"
            ).format(
                schema=sql.Identifier(schema),
                table=sql.Identifier(table_name),
                column=sql.Identifier(column_name),
                sequence=sql.SQL(sequence_name),
            )

            self.run_query(alter_table_query, return_df=False)

            self.logger.info(
                f"Sequence {sequence_name} created successfully and set as default for {table_name}.{column_name}."
            )

        except Exception as e:
            self.logger.error(
                f"Error occurred while creating sequence and setting default: {e}"
            )
            raise e

    def update_sequence(
        self, table_name: str, column_name: str, schema: str = "dpc"
    ) -> None:
        try:
            sequence_name = f"{table_name}_{column_name}_seq"

            update_query = sql.SQL(
                "SELECT setval(%s, (SELECT MAX({}) FROM {}.{}))"
            ).format(
                sql.Identifier(column_name),
                sql.Identifier(schema),
                sql.Identifier(table_name),
            )

            self.run_query(update_query, params=(sequence_name,), return_df=False)

            self.logger.info(
                f"Sequence {sequence_name} updated successfully for table {table_name}."
            )

        except Exception as e:
            self.logger.error(f"Error occurred while updating sequence: {e}")
            raise e

    def get_max_id(
        self, table_name: str, column_name: str = "id", schema: str = "dpc"
    ) -> int:
        """
        Get the maximum id from a specified table.

        Args:
            table_name (str): Table name.
            column_name (str, optional): The name of the id column. Defaults to 'id'.
            schema (str, optional): The schema in which the table resides. Defaults to 'dpc'.

        Returns:
            int: The maximum id from the table.

        Examples:
            >> connector.get_max_id('users', 'id', 'myschema')
        """
        query = sql.SQL("SELECT MAX({column}) FROM {schema}.{table}").format(
            column=sql.Identifier(column_name),
            schema=sql.Identifier(schema),
            table=sql.Identifier(table_name),
        )
        max_id = self.run_query(query, return_df=False, fetch_one=True)
        return max_id

    def check_if_exists(
        self, table_name: str, columns: Dict[str, Any], schema: str = "dpc"
    ) -> bool:
        query = sql.SQL("SELECT EXISTS(SELECT 1 FROM {}.{} WHERE {})").format(
            sql.Identifier(schema),
            sql.Identifier(table_name),
            sql.SQL(" AND ").join(
                sql.SQL("{} = {}").format(sql.Identifier(key), sql.Placeholder())
                for key in columns
            ),
        )
        params = tuple(columns.values())
        return self.run_query(query, params=params, return_df=False, fetch_one=True)

    def update_table(
        self,
        table: str,
        update_cond_dict: dict,
        update_val_dic: dict,
        schema: str = "dpc",
    ):
        """
        Update a record in the specified table with the given conditions and values.

        Args:
            table (str): The name of the table to update.
            update_cond_dict (dict): A dictionary containing the conditions to identify the record(s) to update.
            update_val_dic (dict): A dictionary containing the columns and their new values.
            schema (str, optional): The schema in which the table resides. Defaults to 'dpc'.

        Example:
            >> update_cond_dict = {'id': 42}
            >> update_val_dic = {'name': 'John', 'email': 'john@example.com'}
            >> connector.update_table('users', update_cond_dict, update_val_dic)
        """
        conditions = [
            sql.SQL(" = ").join([sql.Identifier(k), sql.Literal(v)])
            for k, v in update_cond_dict.items()
        ]
        values = [
            sql.SQL(" = ").join([sql.Identifier(k), sql.Literal(v)])
            for k, v in update_val_dic.items()
        ]
        query = sql.SQL("UPDATE {}.{} SET {} WHERE {}").format(
            sql.Identifier(schema),
            sql.Identifier(table),
            sql.SQL(", ").join(values),
            sql.SQL(" and ").join(conditions),
        )
        self.run_query(query, return_df=False)

    def delete_rows_with_condition(
        self, table: str, delete_cond_dict: dict, schema: str = "dpc"
    ):
        conditions = [
            sql.SQL(" = ").join([sql.Identifier(k), sql.Literal(v)])
            for k, v in delete_cond_dict.items()
        ]
        query = sql.SQL("DELETE FROM {}.{} WHERE {}").format(
            sql.Identifier(schema),
            sql.Identifier(table),
            sql.SQL(" and ").join(conditions),
        )
        self.run_query(query, return_df=False)


db_connector = PostgreSQLConnector(**db_config)
