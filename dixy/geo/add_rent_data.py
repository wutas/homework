import pandas as pd
import numpy as np
import psycopg2
import pyodbc
from pyproj import _datadir, datadir

import csv
from io import StringIO
from sqlalchemy import create_engine


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
    pass
# In[19]:

def main():
    print('load real_property')
    conn = pyodbc.connect('Driver={SQL Server};'
                              'Server=SRV-MTADB-LSN;'
                              'Database=SUPMIRA;'
                              'Trusted_Connection=yes;')


    sql =   "Select * from SUPMIRA.dbo.real_property where TypeRent = N'Сдам'  "

    df = (pd.read_sql(sql, conn))

    conn.close()

    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')


    # In[25]:

    print('drop commercial_dixy and test')
    con = psycopg2.connect(dbname='spatial001', user='postgres', password='', host='10.0.28.70')
    sql2 = "DROP TABLE if exists public.test"
    sql3 = "DROP TABLE if exists public.commercial_dixy"

    cur = con.cursor()
    cur.execute(sql2)
    cur.execute(sql3)
    con.commit()
    con.close()
    cur.close()


    # In[27]:



    print('create test')
    engine = create_engine('postgresql+psycopg2://postgres:@10.0.28.70:5432/spatial001')
    df.to_sql('test', engine, method=psql_insert_copy)


    # In[28]:


    print('create commercial_dixy')
    con = psycopg2.connect(dbname='spatial001', user='postgres', password='', host='10.0.28.70')
    sql4 = 'select st_transform(st_setsrid(st_point("Longitude", "Latitude"), 4326), 3857) as geom, * into public.commercial_dixy from public.test'

    cur = con.cursor()
    cur.execute(sql4)
    cur.execute(sql2)
    con.commit()
    con.close()
    cur.close()

if __name__ == '__main__':
    main()

