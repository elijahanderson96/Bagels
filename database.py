
def df_to_postgres(df, table_name, con, schema='fundamentals', replace=False):
    """Wrapper around df.to_sql method
    Args:
        df: dataframe to insert to postgres
        table_name: name of table
        con: connection url
        schema: schema where table is located
        replace: whether to append or replace the table

    Returns: df.to_sql()
    """
    insert_method = 'replace' if replace else 'append'
    return df.to_sql(table_name, con, schema=schema, if_exists=insert_method)
