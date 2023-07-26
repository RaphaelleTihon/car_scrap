import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

np.seterr(all='raise')

japanese_brands = ["toyota", "nissan", "honda"]
american_brands = ["ford", "chevrolet", "dodge"]

#valid data is data where we have at least the price, and the km
def get_valid_data(df):
    print("All data: ", len(df))
    df = df[df["km"] > 0]
    print("Data with km info: ", len(df))
    df = df[df["price"] != "Surdemande"]
    print("Data with price info: ", len(df))
    df["price"] = pd.to_numeric(df["price"])

    return df

def get_data_in_range(df, column, a, b):

    lower_df = df[df[column] < b]
    higher_lower_df = df[df[column] > a]

    return higher_lower_df



if __name__ == "__main__":

    df = pd.read_csv("first_batch.csv")

    toyota_data = df[df["brand"] == "toyota"]
    ford_data = df[df["brand"] == "ford"]

    print(len(toyota_data))
    print(len(ford_data))


    toyota_data = get_valid_data(toyota_data)
    ford_data = get_valid_data(ford_data)

    toyota_km = toyota_data["km"].values
    toyota_price = toyota_data["price"].values
    toyota_price = toyota_price.astype(np.int32)

    ford_km = ford_data["km"].values
    ford_price = ford_data["price"].values
    ford_price = ford_price.astype(np.int32)

    ranges = np.arange(0, 150001, 5000)
    print(ranges)

    ratios_toyota = []
    ratios_ford = []

    initial_prices_toyota = np.mean(get_data_in_range(toyota_data, "km", ranges[0], ranges[1])["price"].values)
    initial_prices_ford = np.mean(get_data_in_range(ford_data, "km", ranges[0], ranges[1])["price"].values)

    print(initial_prices_toyota)
    print(initial_prices_ford)

    for i in range(1, len(ranges)):
        a = ranges[i-1]
        b = ranges[i]

        curr_price_toyota = np.mean(get_data_in_range(toyota_data, "km", a, b)["price"].values)
        curr_price_ford = np.mean(get_data_in_range(ford_data, "km", a, b)["price"].values)

        curr_ratio_toyota = curr_price_toyota/initial_prices_toyota
        curr_ratio_ford = curr_price_ford/initial_prices_ford

        ratios_toyota.append(curr_ratio_toyota)
        ratios_ford.append(curr_ratio_ford)

    plt.plot(np.arange(len(ratios_toyota)), ratios_toyota, color="b")
    plt.plot(np.arange(len(ratios_ford)), ratios_ford, color="r")
    plt.show()
    print(toyota_data[["title", "brand", "model", "submodel", "year"]].head(10))
    not_modelled = toyota_data[toyota_data["model"] == "Not Recognized"]
    print("-------------------------------------------------------------------------------------")
    print(not_modelled.head(10))
    #print(len(ford_data[ford_data["model"] == "F150"]))




    #plt.scatter(ford_km, ford_price, color="r")
    #plt.scatter(toyota_km, toyota_price, color="b")
    #plt.title("prix en fonction du kilometrage")
    #plt.xlabel("kilometres")
    #plt.ylabel("price")
    #plt.show()
    