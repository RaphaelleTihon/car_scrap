from scraper import get_kijiji_ads
from db_interface import add_df_values_to_db, get_size_db
import time
import logging

def add_df_to_db(con, df):

    cur = con.cursor()

def main_loop():

    while True:
        df = get_kijiji_ads(1)
        add_df_values_to_db(df)        
        size_db = get_size_db()
        logging.info(f"Current size of database: {size_db}")
        time.sleep(300)


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    main_loop()