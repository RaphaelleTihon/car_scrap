import sqlite3
#con = sqlite3.connect("tuto")
#cur = con.cursor()
#ur.execute("""CREATE TABLE car(
#            brand, 
#            model, 
#            year, 
#            km, 
#            price,
#            id NOT NULL PRIMARY KEY
#            )""")
#cur.execute("CREATE TABLE movie(title, year, score)") #throws an exception if executed and table already exists

#cur.execute("SELECT name FROM sqlite_master")
#print(cur.fetchall())

#Strings to be inserted need to be quoted
#cur.execute("""
#    INSERT into car VALUES
#    ('toyota', 'yaris', 2015, 20000, 13000, 1),
#    ('ford', 'f150', 2023, 2000, 90000, 2)
#""")
#con.commit()
#con.commit() An insert is a transaction, all transactions have to be commited

#cur.execute("SELECT brand, id FROM car WHERE id=3")
#print(cur.fetchall())
#cur.execute("SELECT brand, model, year FROM car")
#print(cur.fetchall())

#data = [
#    ("Monty Python Live at the Hollywood Bowl", 1982, 7.9),
#    ("Monty Python's The Meaning of Life", 1983, 7.5),
#    ("Monty Python's Life of Brian", 1979, 8.0),
#]

#cur.executemany("INSERT into movie VALUES(?,?,?)", data)
#con.commit()

#cur.execute("SELECT * FROM movie")
#print(cur.fetchall())

#con.close()

con = sqlite3.connect("database/kijiji_car_db")
cur = con.cursor()

cur.execute("SELECT name FROM sqlite_master")
print(cur.fetchall())


cur.execute(f"SELECT brand, model FROM car WHERE brand = 'toyota'")
print(len(cur.fetchall()))