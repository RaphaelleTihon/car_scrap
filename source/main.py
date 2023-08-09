from scraper import get_kijiji_ads, reparse_title
from db_interface import add_df_values_to_db, get_size_db, get_all, update_entry
from setup_db import car_entry_columns
import time
import logging
import pandas as pd
import argparse

def add_df_to_db(con, df):

    cur = con.cursor()

def main_loop():

    while True:
        df = get_kijiji_ads(1)
        add_df_values_to_db(df)        
        size_db = get_size_db()
        logging.info(f"Current size of database: {size_db}")
        time.sleep(300)

def reparse_title_and_override_db():
    all_data = get_all()
    df = pd.DataFrame(all_data, columns = car_entry_columns)
    updates_df = reparse_title(df)
    logging.info(f"Number of entries in db: {len(updates_df)}")
    updates_df = updates_df[updates_df["is_modified"] == 1]
    logging.info(f"Number of entries with updated information: {len(updates_df)}")
    updates_df.to_csv("csv_data/test_updates.csv")
    update_entry(updates_df, ["title", "brand", "model", "submodel"])

    #cursor.execute('''UPDATE books SET price = ? WHERE id = ?''', (newPrice, book_id))

def parse_args():

    parser = argparse.ArgumentParser(
                    prog='car_scrap',
                    description='Kijiji automated scraper',
                    epilog='Bai bai')
    
    action_choices = ["mainloop", "reparse"]
    parser.add_argument("-a", "--action", choices = action_choices, default = "mainloop")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    action = args.action
    if action == "mainloop":
        main_loop()
    elif action == "reparse":
        reparse_title_and_override_db()