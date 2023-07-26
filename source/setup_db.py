import sqlite3
#GOLDEN VALUES FOR DATABASE COLUMNS
#The primary key, the url, must be the last entry
car_entry_columns = ["title",
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
]

def create_db():
    con = sqlite3.connect("database/kijiji_car_db")
    cur = con.cursor()

    table_column_strings = ""
    for i in range(len(car_entry_columns)-1):
        table_column_strings += f"{car_entry_columns[i]},\n"

    table_column_strings += f"{car_entry_columns[-1]} NOT NULL PRIMARY KEY"
    

    table_columns = ""

    cur.execute(f"CREATE TABLE car({table_column_strings})")

if __name__ == "__main__":

    create_db()