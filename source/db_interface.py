import sqlite3
from setup_db import car_entry_types
import logging
import pandas as pd

con = sqlite3.connect("database/kijiji_car_db")
cur = con.cursor()

def add_df_values_to_db(df):
    list_string_values = []

    row_tuples = [ tuple([v for k, v in row.items()]) for ix, row in df.iterrows() ]

    """def get_row_strings(row): OLD IMPLEMENTATION, TO REMOVE AFTER LENGTHY SUCCESSFUL TESTING OF NEW ONE

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

    logging.debug(insert_string)"""

    seen_urls = []
    final_entry_tuples = []
    
    for entry_tuple in row_tuples:
        
        #The db is poorly structured, the url, the primary, should be the first column so that adding columns does not mess up indexes
        url = entry_tuple[-2]
        cur.execute(f"SELECT brand, id FROM car WHERE url='{url}'")
        
        if len(cur.fetchall()) == 0 and url not in seen_urls:
            
            final_entry_tuples.append(entry_tuple)
            #sometimes the same ad can appear twice on a page, 
            #this checks and prevents double insertions in that case
            seen_urls.append(url) 

        
    number_columns_string=f"({','.join(['?']*len(car_entry_types))})"
    insert_string = f"INSERT into car VALUES {number_columns_string}"

    try:
        cur.executemany(insert_string, final_entry_tuples)
        con.commit()
    except Exception as e:
        print(insert_string)
        print(final_entry_tuples)
        raise(e)

def get_size_db():
 
    cur.execute("SELECT COUNT(*) FROM car")
    return cur.fetchall()[0][0]

def get_brand_model_entries(brand, model):

    cur.execute(f"SELECT * FROM car WHERE brand='{brand}' AND model='{model}'")
    return cur.fetchall()

def get_all():
    
    cur.execute("SELECT * FROM car")
    return cur.fetchall()

def update_entry(df, columns_to_update):
    """Columns to update must be a list of columns present in the df AND in the table"""

    columns_to_update.append("url") #adding url to be part of the tuple
    df = df[columns_to_update]
    row_tuples = [ tuple([v for k, v in row.items()]) for ix, row in df.iterrows() ] #creating tuples to be added

    columns_to_update.pop() #do not update url

    update_string = "UPDATE car SET "
    for column in columns_to_update:
        update_string += f"{column} = ?,"
    update_string = update_string[:-1] #remove last comma
    update_string += "WHERE url = ?"

    try:
        cur.executemany(update_string, row_tuples)
        con.commit()
    except Exception as e:
        print(update_string)
        print(row_tuples)
        raise(e)
    
    pass