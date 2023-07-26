import sqlite3
from setup_db import car_entry_types
import logging
import pandas as pd

con = sqlite3.connect("database/kijiji_car_db")
cur = con.cursor()

def add_df_values_to_db(df):
    list_string_values = []

    def get_row_strings(row):

        row_list = row.values.flatten().tolist()
        row_string = "("

        for i in range(len(car_entry_types)-1):

            if car_entry_types[i] == "str":
                row_string += f"'{row_list[i]}',"

            elif car_entry_types[i] == "num":
                row_string += f"{row_list[i]},"
        
        #adding the last entry, should always be the primary key the url
        row_string += f"'{row_list[-1]}')"

        return pd.Series([row_string, row["url"]], index=["string_value", "url"])
    
    list_string_values = pd.DataFrame(df.apply(lambda row: get_row_strings(row), axis=1))
    #print(list_string_values)
    #print(list_string_values.head())
    string_values = list_string_values["string_value"]
    urls = list_string_values["url"]
    final_list = []
    to_insert_urls = []
    for i, value in enumerate(string_values):
        cur.execute(f"SELECT brand, id FROM car WHERE url='{urls[i]}'")
        if len(cur.fetchall()) == 0 and urls[i] not in to_insert_urls:
            final_list.append(value)
            to_insert_urls.append(urls[i])

    if len(final_list) == 0:
        logging.info("No new ad to add at this time")
        return

    insert_string = "INSERT into car VALUES\n"
    for i in range(len(final_list)-1):
        insert_string += f"{final_list[i]},\n"

    insert_string += f"{final_list[-1]}"

    logging.debug(insert_string)
    try:
        cur.execute(insert_string)
        con.commit()
    except Exception as e:
        print(insert_string)
        raise(e)

def get_size_db():
 
    cur.execute("SELECT COUNT(*) FROM car")
    return cur.fetchall()[0][0]