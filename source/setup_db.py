import sqlite3
import pandas as pd
#import db_interface
#GOLDEN VALUES FOR DATABASE COLUMNS
#The primary key, the url, must be the last entry
car_entry_columns = [
    "title",
    "id",
    "brand",
    "model",
    "submodel",
    "power",
    "year",
    "km",
    "transmission",
    "price",
    "day_posted",
    "month_posted",
    "year_posted",
    "hour_posted",
    "minute_posted",
    "time_until_unavailable",
    "url",
    "tracking"
]

#Note: -1 is equivalent to Unknown for numerical entries

car_entry_types = [
    "str", #title
    "num", #id
    "str", #brand
    "str", #model
    "str", #submodel
    "str", #power
    "num", #year
    "num", #km
    "str", #transmission
    "num", #price some ads have "surdemnande" which is transformed into -1
    "num", #day posted
    "num", #month posted
    "num", #year posted
    "num", #hour posted
    "num", #minute posted
    "num", #time until unavailable
    "str", #url
    "num", #tracking
]

MODEL_TRACKED_LIST = []

def convert_df_types(df):

    for i, column in enumerate(car_entry_columns):

        if car_entry_types[i] == "num":

            df[column] = pd.to_numeric(df[column])

    return df

def create_kijiji_db():
    con = sqlite3.connect("database/kijiji_car_db")
    cur = con.cursor()

    table_column_strings = ""
    for i in range(len(car_entry_columns)):
        if car_entry_columns[i] == "url":
            table_column_strings += f"{car_entry_columns[i]} NOT NULL PRIMARY KEY,"
        else:
            table_column_strings += f"{car_entry_columns[i]},\n"
        
    table_column_strings = table_column_strings[:-1]

    print(table_column_strings)

    cur.execute(f"CREATE TABLE car({table_column_strings})")

    a = cur.execute("PRAGMA table_info('car')")

    for i in a:

        print(i)

def add_column():

    columns_to_add = ["tracking"]

    con = sqlite3.connect("database/kijiji_car_db")
    cur = con.cursor()

    a = cur.execute("PRAGMA table_info('car')")
    for column_info in a:
        #it's the name of the column
        if column_info[1] in columns_to_add:
            raise Exception(f"Tried adding {columns_to_add} but column {column_info[1]} already exists in the database")
        
    for column in columns_to_add:
        cur.execute(f"alter table car add column {column} ")


if __name__ == "__main__":

    add_column()
    con = sqlite3.connect("database/kijiji_car_db")
    cur = con.cursor()
    a = cur.execute("PRAGMA table_info('car')")
    for column_info in a:
        #it's the name of the column
        print(column_info)

    url_test = "/v-autos-camions/ville-de-montreal/2021-toyota-corolla-le-upgrade-mags-toit-ouvrant/m6060724"

    #quer = cur.execute(f"SELECT * FROM car WHERE url='{url_test}'")
    #print(quer.fetchall())
    #cur.execute(f"UPDATE car SET tracking =  WHERE url = '{url_test}'")
    #con.commit()