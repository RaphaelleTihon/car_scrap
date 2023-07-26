import requests
from bs4 import BeautifulSoup as souper
import re
import pandas as pd
import car_info as car_info
from datetime import datetime, timedelta
from setup_db import car_entry_columns
import logging
import pytz

hybrid_option = ["hybrid", "hybride"]
electric_option = ["electrique", "électrique", "éléctrique", "electric"]

def clean_string(string):

    string = string.replace(" ", "")
    string = string.replace(chr(10), "")
    return string
#----------------------------------------------------------------------------------------------------------------------

def parse_details(detail_string):
    
    detail_string = clean_string(detail_string)

    transmission = "Unknown"
    km = -1
    km_string = None
    if "Automatique" in detail_string:
        transmission = "A"
        detail_string.replace("Automatique", "")
    elif "Manuelle" in detail_string:
        transmission = "M"
        detail_string.replace("Manuelle", "")

    if "|" in detail_string:
        all_strings = detail_string.split("|")
        for string in all_strings:
            if "km" in string:
                km_string = string
        
    elif "km" in detail_string:
        km_string = detail_string

    if km_string is not None:
        km_string = clean_string(km_string)
        km_string = km_string.replace("km", "")
        km_string = km_string.replace(chr(160), "")
        #print(ord(km_string[3]))

        km = int(km_string)

    return transmission, km
#----------------------------------------------------------------------------------------------------------------------

def parse_title(title):
    
    title = title.lower().strip()
    title = title.replace("\"", "")
    title = title.replace("'", "")
    title = title.replace("!", "")
    title_brand = "Not Recognised"
    done = False
    for brand in car_info.brand_dict.keys():

        for brand_name in car_info.brand_dict[brand]:

            if brand_name in title:
                title_brand = brand
                done = True
                break

        if done:
            break

    return title, title_brand
#----------------------------------------------------------------------------------------------------------------------

def get_brand_models_reparse(row, brand = None):
    
    model = row["model"]
    if row["brand"] != brand:
        return model
    
    brand_models = car_info.brand_to_models[brand]
    title = row["title"]
    
    for key in brand_models.keys():
        for model_name in brand_models[key]:
            if model_name in title:
                model = key
                #logging.debug(model)
                break
    return model
#----------------------------------------------------------------------------------------------------------------------

def get_brand_models(title, brand = None):
    
    model = "Not Recognized"

    if brand not in car_info.brand_to_models.keys():
        return model
    
    brand_models = car_info.brand_to_models[brand]
    
    for key in brand_models.keys():
        for model_name in brand_models[key]:
            if model_name in title:
                model = key
                #logging.debug(model)
                return model
    return model
#----------------------------------------------------------------------------------------------------------------------

def get_submodel(title, brand, model):
    
    sub_model = "Not Recognized"

    if brand not in car_info.brand_models_to_subs.keys():
        return sub_model
    
    brand_models_to_sub = car_info.brand_models_to_subs[brand]

    if model not in brand_models_to_sub.keys():
        return sub_model

    model_to_subs = brand_models_to_sub[model]

    for key in model_to_subs.keys():
        for sub_name in model_to_subs[key]:
            if sub_name in title:
                sub_model = key
                #logging.debug(sub_model)
                return sub_model
    return sub_model
#----------------------------------------------------------------------------------------------------------------------

def parse_price(price):

    price = price.replace("$", "")
    price = price.replace(" ", "")
    price = price.replace(chr(160), "")
    price = price.replace("\n", "")

    try:
        price = int(price.split(",")[0])
    except:
        pass

    if price == "Surdemande":
        return -1
    elif price == "Gratuit":
        return -2

    return int(price)
#----------------------------------------------------------------------------------------------------------------------

months_dict = {
    "janvier" : 1,
    "février" : 2,
    "mars" : 3,
    "avril" : 4,
    "mai" : 5,
    "juin" : 6,
    "juillet" : 7,
    "août" : 8,
    "septembre" : 9,
    "octobre" : 10,
    "novembre" : 11,
    "décembre" : 12
}

def parse_date(date):

    if "-" not in date:
        now = datetime.now(pytz.timezone("US/Eastern"))
        hours_delta = 0
        minute_delta = 0
                    
        delta_search = re.search("[0-9]{1,2}", date)
        delta = -1
        if delta_search is not None:
            delta = delta_search.group()
            delta = int(delta)
        if "minute" in date:
            minute_delta = delta
        elif "heure" in date:
            hours_delta = delta

        time_delta = timedelta(hours = hours_delta, minutes = minute_delta)
        date_posted = now - time_delta

        return date_posted.day, date_posted.month, date_posted.year, date_posted.hour, date_posted.minute
        
    else:
        date = date.split("-")
        day = date[0]
        month = months_dict[date[1]]
        year = date[2]

    return day, month, year, -1, -1
#----------------------------------------------------------------------------------------------------------------------    

def get_kijiji_ads(n=1):
    """
    Returns the first n pages of kijiji car ads by date added as a dataframe
    by default n=1
    """

    #https://www.kijiji.ca/b-autos-camions/ville-de-montreal/used/page-3/c174l1700281a49
    first = "https://www.kijiji.ca/b-cars-trucks/ville-de-montreal/c174l1700281?sort=dateDesc"
    #iter_on = f"https://www.kijiji.ca/b-autos-camions/ville-de-montreal/new__used/page-{num}/c174l1700281a49"
    request_url = first

    data = { column : [] for column in car_entry_columns}
    for page_num in range(1):

        page=requests.get(request_url)
        soup=souper(page.content, "html.parser")

        all_items=soup.select("div.search-item")

        for i in range(len(all_items)):
            item = all_items[i]
            id = item.attrs["data-listing-id"]
            item_url = item.attrs["data-vip-url"]
            #Some ids are 7 char longs, they need a prefix, m works, maybe for montreal? Need further info
            #ids of the form 1665942300, 10 char long, work as is
            title = item.select("a.title")[0].text 
            title, brand = parse_title(title)
            model = get_brand_models(title, brand)
            sub_model = get_submodel(title, brand, model)
            price = item.select("div.price")[0].text
            details = item.select("div.details")[0].text
            date = item.select("span.date-posted")[0].text
            day_posted, month_posted, year_posted, hour_posted, minute_posted = parse_date(date)

            title = title.replace(chr(10), "")
            price = parse_price(price)
            details = parse_details(details)

            power = "Combustion"
            for hybrid in hybrid_option:
                if hybrid in title:
                    power = "Hybrid"

            for elec in electric_option:
                if elec in title:
                    power = "Electric"

            #logging.debug(title)
            #logging.debug(brand)
            #logging.debug(id)
            #logging.debug(price)
            #logging.debug(details)

            year_search = re.search("20[0-9]{2}", title)
            year = -1
            if year_search is not None:
                year = year_search.group()
            #logging.debug(year)

            data["title"].append(title)
            data["id"].append(id)
            data["brand"].append(brand)
            data["model"].append(model)
            data["submodel"].append(sub_model)
            data["power"].append(power)
            data["year"].append(year)
            data["km"].append(details[1])
            data["transmission"].append(details[0])
            data["price"].append(price)
            data["day_posted"].append(day_posted)
            data["month_posted"].append(month_posted)
            data["year_posted"].append(year_posted)
            data["hour_posted"].append(hour_posted)
            data["minute_posted"].append(minute_posted)
            data["time_until_unavailable"].append(-1)
            data["url"].append(item_url)

        request_url = f"https://www.kijiji.ca/b-cars-trucks/ville-de-montreal/page-{page_num+2}/c174l1700281?sort=dateDesc"

    df = pd.DataFrame(data)
    return df

#----------------------------------------------------------------------------------------------------------------------

def reparse():
    
    df = pd.read_csv("first_batch.csv")
    
    df["model"] = df.apply(lambda row: get_brand_models_reparse(row, brand="toyota"), axis=1)
    df["model"] = df.apply(lambda row: get_brand_models_reparse(row, brand="ford"), axis=1)
    df.to_csv("first_batch.csv", index=False)


df = get_kijiji_ads(1)
#reparse()
#reparse_clean()