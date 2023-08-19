import scraper
from db_interface import (
    add_df_values_to_db,
    get_size_db,
    get_all,
    update_entry,
    get_all_tracking,
)
from setup_db import car_entry_columns
import time
import logging
import pandas as pd
import argparse
from threading import Thread


def main_loop():
    n = 0

    def scrape_loop():
        while True:
            scrape()
            time.sleep(180)

    def track_loop():
        while True:
            track()
            time.sleep(3600)

    t_scrape = Thread(target=scrape_loop)
    t_scrape.start()

    t_track = Thread(target=track_loop)
    t_track.start()


def scrape():
    df = scraper.get_kijiji_ads(1)
    add_df_values_to_db(df)
    size_db = get_size_db()
    logging.info(f"Current size of database: {size_db}")


def track():
    all_tracking = get_all_tracking()
    df = pd.DataFrame(all_tracking, columns=car_entry_columns)
    df = scraper.track(df)
    update_entry(df, columns_to_update=["time_until_unavailable", "tracking"])


def reparse_title_and_override_db():
    all_data = get_all()
    df = pd.DataFrame(all_data, columns=car_entry_columns)
    updates_df = scraper.reparse_title(df)
    logging.info(f"Number of entries in db: {len(updates_df)}")
    updates_df = updates_df[updates_df["is_modified"] == 1]
    logging.info(f"Number of entries with updated information: {len(updates_df)}")
    # updates_df.to_csv("csv_data/test_updates.csv")
    update_entry(updates_df, ["title", "brand", "model", "submodel"])

    # cursor.execute('''UPDATE books SET price = ? WHERE id = ?''', (newPrice, book_id))


def parse_args():
    parser = argparse.ArgumentParser(
        prog="car_scrap", description="Kijiji automated scraper", epilog="Bai bai"
    )

    action_choices = ["mainloop", "reparse"]
    parser.add_argument("-a", "--action", choices=action_choices, default="mainloop")

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

if __name__ == "__main__2":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    all_tracking = get_all_tracking()
    df = pd.DataFrame(all_tracking, columns=car_entry_columns)
    df.to_csv("csv_data/all_tracking.csv")
