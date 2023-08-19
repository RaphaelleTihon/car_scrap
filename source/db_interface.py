import sqlite3
from setup_db import car_entry_types
import logging
import pandas as pd


def add_df_values_to_db(df):
    con = sqlite3.connect("database/kijiji_car_db")
    cur = con.cursor()

    list_string_values = []

    row_tuples = [tuple([v for k, v in row.items()]) for ix, row in df.iterrows()]

    seen_urls = []
    final_entry_tuples = []

    for entry_tuple in row_tuples:
        # The db is poorly structured, the url, the primary, should be the first column so that adding columns does not mess up indexes
        url = entry_tuple[-2]
        cur.execute(f"SELECT brand, id FROM car WHERE url='{url}'")

        if len(cur.fetchall()) == 0 and url not in seen_urls:
            final_entry_tuples.append(entry_tuple)
            # sometimes the same ad can appear twice on a page,
            # this checks and prevents double insertions in that case
            seen_urls.append(url)

    number_columns_string = f"({','.join(['?']*len(car_entry_types))})"
    insert_string = f"INSERT into car VALUES {number_columns_string}"

    try:
        cur.executemany(insert_string, final_entry_tuples)
        con.commit()
    except Exception as e:
        print(insert_string)
        print(final_entry_tuples)
        raise (e)


def get_size_db():
    con = sqlite3.connect("database/kijiji_car_db")
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM car")
    return cur.fetchall()[0][0]


def get_brand_model_entries(brand, model):
    con = sqlite3.connect("database/kijiji_car_db")
    cur = con.cursor()
    cur.execute(f"SELECT * FROM car WHERE brand='{brand}' AND model='{model}'")
    return cur.fetchall()


def get_all():
    con = sqlite3.connect("database/kijiji_car_db")
    cur = con.cursor()
    cur.execute("SELECT * FROM car")
    return cur.fetchall()


def get_all_tracking():
    con = sqlite3.connect("database/kijiji_car_db")
    cur = con.cursor()
    cur.execute("SELECT * FROM car WHERE tracking = 1")
    return cur.fetchall()


def update_entry(df, columns_to_update):
    """Columns to update must be a list of columns present in the df AND in the table"""

    con = sqlite3.connect("database/kijiji_car_db")
    cur = con.cursor()

    columns_to_update.append("url")  # adding url to be part of the tuple
    df = df[columns_to_update]
    row_tuples = [
        tuple([v for k, v in row.items()]) for ix, row in df.iterrows()
    ]  # creating tuples to be added

    columns_to_update.pop()  # do not update url

    update_string = "UPDATE car SET "
    for column in columns_to_update:
        update_string += f"{column} = ?,"
    update_string = update_string[:-1]  # remove last comma
    update_string += "WHERE url = ?"

    try:
        cur.executemany(update_string, row_tuples)
        con.commit()
    except Exception as e:
        print(update_string)
        print(row_tuples)
        raise (e)

    pass
