import csv
from io import StringIO

import pandas as pd
import psycopg2
from psycopg2 import sql
from sqlalchemy import create_engine


class PostgresDB:
    '''
    # Create a table
        db = PostgresDB(host="172.29.112.1", port="5432", user="elijah", password="Poodle!3", dbname="market_data")
        db.connect()

        table_name = "test_table"
        columns = {
            "id": "SERIAL",
            "name": "VARCHAR(255) NOT NULL",
            "age": "INTEGER"
        }

        db.run_query('DROP TABLE if exists test_table CASCADE;')

        db.create_table(table_name, columns)

        # Insert data into the table
        db.run_query("SELECT * FROM test_table;")

        # Add primary key
        db.modify_table(table_name, "add_primary_key", column="id", constraint_name="test_table_pkey")

        # Add unique key
        db.modify_table(table_name, "add_unique_key", columns=["name", "age"], constraint_name="test_table_unique")

        # Add foreign key
        db.create_table("reference_table", {"ref_id": "SERIAL PRIMARY KEY"})
        db.modify_table(table_name, "add_foreign_key", column="age", reference_table="reference_table",
                        reference_column="ref_id", constraint_name="test_table_fk")

        # Add sequence
        db.run_query("CREATE SEQUENCE test_table_age_seq;")
        db.modify_table(table_name, "add_sequence", column="age", sequence_name="test_table_age_seq")

        # Disconnect from the database
        db.disconnect()
        '''
    def __init__(self, host, port, user, password, dbname):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.dbname = dbname
        self.conn = None

    def connect(self):
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                dbname=self.dbname
            )
            print("Connection successful")
        except Exception as e:
            print("Error connecting to the database: ", e)

    def disconnect(self):
        if self.conn is not None:
            self.conn.close()
            print("Connection closed")

    def run_query(self, query, params=None):
        with self.conn.cursor() as cur:
            try:
                cur.execute(query, params)
                if query.lower().startswith("select"):
                    data = cur.fetchall()
                    column_names = [desc[0] for desc in cur.description]
                    df = pd.DataFrame(data, columns=column_names)
                    print(f'Returned dataframe of shape: {df.shape[0]} x {df.shape[1]}')
                    return df
                else:
                    self.conn.commit()
                    print("Query executed successfully")
            except Exception as e:
                print("Error executing query: ", e)

    def create_table(self, table_name, columns):
        with self.conn.cursor() as cur:
            try:
                columns_sql = ", ".join([f"{col} {data_type}" for col, data_type in columns.items()])
                create_table_query = sql.SQL("CREATE TABLE {} ({});").format(
                    sql.Identifier(table_name),
                    sql.SQL(columns_sql)
                )
                cur.execute(create_table_query)
                self.conn.commit()
                print(f"Table '{table_name}' created successfully")
            except Exception as e:
                print("Error creating table: ", e)

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


db_connector = PostgresDB(host='172.18.144.1',
                          user='elijah',
                          dbname='market_data',
                          port='5432',
                          password='Poodle!3')

db_connector.connect()