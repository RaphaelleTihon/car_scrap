from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from db_interface import get_brand_model_entries
import pandas as pd
from setup_db import car_entry_columns
import matplotlib.pyplot as plt
import numpy as np
from setup_db import convert_df_types
import car_info as car_info

class MultiPolyRegr():
    """
    Regresses a function of the form y = k*x^(n) with 0.1 < n < 3
    x can be of dimension higher than 1, n will be calculated for each dimension
    individually
    """


    def __init__(self):
        #Value used to scaled the data to a linear relationship
        self.mins = []
        self.coeffs = []

    def find_scaling_coeff(self, x, y):
        """
        find the best exponent value to scale the x data so that it's linearly correlated with y
        the search done here supposes that the linear r2 coefficient is concave, meaning
        that there is a global minimum and no local minima 
        Here x is one dimensional
        """

        minimum = np.min(x)
        x_norm = x-minimum

        coeffs = np.arange(0.1, 3, 0.05)

        r2_scores = []
        best_score = 0
        best_coeff = 1
        for coeff in coeffs:

            scaled_x = (x_norm**(coeff)).reshape(-1, 1)
            reg = LinearRegression().fit(scaled_x, y)
            r2 = reg.score(scaled_x, y)
            r2_scores.append(r2)
            if r2 > best_score:
                best_score = r2
                best_coeff = coeff

        return minimum, best_coeff, coeffs, r2_scores
    
    def generate_scalings(self, x, y):

        n_dims_to_scale = x.shape[1]
        for i in range(n_dims_to_scale):

            #ith feature vector
            x_i = x[:, i]
            minimum, coeff, _, _ = self.find_scaling_coeff(x_i, y)
            self.mins.append(minimum)
            self.coeffs.append(coeff)

    def scale(self, x):
        n_dims_to_scale = x.shape[1]
        scaled_x = np.zeros(x.shape)

        for i in range(n_dims_to_scale):
            x_i = x[:, i]
            scaled_x_i = x_i-self.mins[i]
            scaled_x_i = np.where(scaled_x_i > 0, scaled_x_i, 1)
            scaled_x_i = (scaled_x_i)**(self.coeffs[i])
            scaled_x[:, i] = scaled_x_i.flatten() 
    
        return scaled_x

    def fit(self, x, y):

        assert len(x.shape) == 2

        self.generate_scalings(x, y)
        scaled_x = self.scale(x)    

        self.regr = LinearRegression().fit(scaled_x, y)

    def predict(self, x):
        #print(self.coeffs)
        #print(self.mins)
        scaled_x = self.scale(x)
        prediction = self.regr.predict(scaled_x)
        return prediction
#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------

class LinRegr:

    def __init__(self):
        pass

    def fit(self, x, y):
        self.regr = LinearRegression().fit(x, y)

    def predict(self, x):
        return self.regr.predict(x)
#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------

class Scaler:
    """
    for this model the first dim of x are the base values 
    and the following dimensions are one hot encoded categories 
    this scales the base values closer to the category specific mean
    """
    def __init__(self):
        
        self.coeffs = []


    def fit(self, x, y):

        n = x.shape[1]-1 #removes the base value dim
        base_values = x[:, -1]
        for i in range(n):
            category = x[:, i+1]
            arr = np.array([base_values, y, category]).transpose()
            #print(arr)
            filtered_x = arr[arr[:, 2] > 0.5]
            print(len(filtered_x))
            mpe = mean_percentage_error(filtered_x[:, 1], filtered_x[:, 0])
            self.coeffs.append(mpe)
            print(f"mpe for category {i}, number of values: {len(filtered_x)}, mpe: {mpe}")
        print(self.coeffs)
        
    def predict(self, x):

        predictions = []
        for i, value in enumerate(x):
            base_value = value[-1]
            index = np.argmax(value[:-1])
            coeff = self.coeffs[index]
            predictions.append(base_value*(1+coeff))

        return np.array(predictions)


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------

class LayeredModel():
    """Infers"""
    def __init__(self, models, indexes):
        """
        models are a list of models for example [LinRegr, PolyRegr, Scaler]
        indexes are a list of index list, saying which columns of the feature go with which models example [[0], [1, 2], [3]]
        """
        self.models = models
        self.indexes = indexes
        self.instanced_models = []

    def get_data_subslice(self, index_slicer, x, extra=False):

        if extra:
            x_subslice = np.zeros([len(x), len(index_slicer)+1])
        else:
            x_subslice = np.zeros([len(x), len(index_slicer)])

        for j, index in enumerate(index_slicer):
            x_subslice[:, j] = x[:, index]
        return x_subslice

    def mainloop(self, x, y=None):
        previous_predictions = None
        for i, model in enumerate(self.models):

            index_slicer = self.indexes[i]
            if i == 0:
                x_model = self.get_data_subslice(index_slicer, x)
            else:
                x_model = self.get_data_subslice(index_slicer, x, extra=True)
                x_model[:, -1] = previous_predictions.flatten()
            
            if y is not None:
                self.instanced_models.append(model())
                self.instanced_models[i].fit(x_model, y)
            
            predictions = self.instanced_models[i].predict(x_model)
            previous_predictions = predictions
        return predictions

    def fit(self, x, y):
        self.mainloop(x, y)

    def predict(self, x):
        return self.mainloop(x)
            
#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------


class LayeredLinRegr():

    def __init__(self):
        print("Assumes that 'year' feature is the first in the feature vector")

    def fit(self, X, y): #receives X features that have already been scaled to be linear with y

        years = X[:, 0].reshape(-1, 1)
        self.reg1 = LinearRegression().fit(years, y)

        pred_layer1 = self.reg1.predict(years)

        print("training R2 of first layer: ", self.reg1.score(years, y))
        print("training MAPE of first layer: ", mean_absolute_percentage_error(y, pred_layer1))

        other_features = X[:, 1:]
        layer1_features = np.zeros([len(other_features), len(other_features[1])+1])
        layer1_features[:, 1:] = other_features
        layer1_features[:, 0] = pred_layer1

        self.reg2 = LinearRegression().fit(layer1_features, y)

        pred_layer2 = self.reg2.predict(layer1_features)
        print("training R2 of second layer: ", self.reg2.score(layer1_features, y))
        print("training MAPE of second layer: ", mean_absolute_percentage_error(y, pred_layer2))
#----------------------------------------------------------------------------------------------------------------------

    def predict_and_score(self, X, y):

        years = X[:, 0].reshape(-1, 1)
        pred_layer1 = self.reg1.predict(years)
        other_features = X[:, 1:]
        layer1_features = np.zeros([len(other_features), len(other_features[1])+1])
        layer1_features[:, 1:] = other_features
        layer1_features[:, 0] = pred_layer1
        pred_layer2 = self.reg2.predict(layer1_features)

        print("testing R2 of second layer: ", self.reg2.score(layer1_features, y))
        print("testing MAPE of second layer: ", mean_absolute_percentage_error(y, pred_layer2))

        return pred_layer2
#----------------------------------------------------------------------------------------------------------------------

def mean_percentage_error(real_values, prediction):

    percentage_errors = (real_values - prediction)/real_values
    return np.mean(percentage_errors)

def find_scaling(x, y):
    """
    find the best exponent value to scale the x data so that it's linearly correlated with y
    the search done here supposes that the linear r2 coefficient is concave, meaning
    that there is a global minimum and no local minima 
    """

    minimum = np.min(x)
    x_norm = x-minimum

    coeffs = np.arange(0.1, 3, 0.05)

    r2_scores = []

    for coeff in coeffs:

        scaled_x = (x_norm**(coeff)).reshape(-1, 1)
        reg = LinearRegression().fit(scaled_x, y)
        r2 = reg.score(scaled_x, y)
        print(f"Score with coeff {coeff}: {r2}")
        r2_scores.append(r2)


def get_valid_data(df):
    print("All data: ", len(df))
    df = df[df["km"] > 0]
    print("Data with km info: ", len(df))
    df = df[df["price"] > 1000] #remove dumb ads with prices of 1
    print("Data with price and km info: ", len(df))
    df = df[df["year"] > 0]
    print("Data with year, price and km info:", len(df))

    return df

def get_year_km_feature(df):

    years = df["year"].values
    kms = df["km"].values
    arr = np.array([years, kms])

    return arr.transpose()

def get_submodel_category_feature(df, brand, model):

    submodel_dict = car_info.brand_models_to_subs[brand][model]
    number_of_submodels = len(submodel_dict.keys())+1
    categories = np.zeros([len(df), number_of_submodels])
    
    def apply_category(row):

        submodel = row["submodel"]
        i = 0
        for key in submodel_dict.keys():
            if submodel == key:
                break
            i+=1
        return pd.Series([i], index=["category"])

    raw_category = pd.DataFrame(df.apply(lambda row: apply_category(row), axis=1))["category"].values

    for i, category in enumerate(raw_category):
        categories[i][category] = 1

    return categories


def get_objective(df):

    prices = df["price"].values
    return np.array(prices)

def layered_regressor(features, objectives):
    X_train, X_test, y_train, y_test = train_test_split(features, prices, random_state=1)
    
def apply_on_submodels(df_array, labels, full_df):

    for i, df in enumerate(df_array):

        prices = df["price"].values
        years = df["year"]
        print(f"mean price for submodels {labels[i]}: {np.mean(prices):.0f}, number of entries {len(prices)}, average year: {np.mean(years):.2f}")

    years = list(range(2006, 2024))

    for year in years:
        year_corolla = full_df[full_df["year"] == year]
        print(f"mean price for {year} corollas: {np.mean(year_corolla['price'].values):.0f}, number of entries {len(year_corolla)}")

def get_all_model_data(brand, model):
    all_data = get_brand_model_entries(brand, model)
    df = pd.DataFrame(all_data, columns = car_entry_columns)
    df = convert_df_types(df)
    df = get_valid_data(df)

    #df.sort_values(by=["price"], ascending=True, inplace=True)

    features = get_year_km_feature(df)
    prices = get_objective(df)
    categories = get_submodel_category_feature(df, brand, model)
    new_features = np.zeros([len(features), features.shape[1]+categories.shape[1]])
    new_features[:, :2] = features
    new_features[:, 2:] = categories
    features = new_features
    return df, features, prices


if __name__ == "__main__2":

    df, features, prices = get_all_model_data("toyota", "corolla")

    ce_corolla_df = df[df["submodel"] == "ce"]
    se_corolla_df = df[df["submodel"] == "se"]
    le_corolla_df = df[df["submodel"] == "le"]
    NR_corolla_df = df[df["submodel"] == "Not Recognized"]

    apply_on_submodels([ce_corolla_df, se_corolla_df, le_corolla_df, NR_corolla_df], ["ce", "se", "le", "NR"], df)

    #print(len(prices))
    #print(features)

    kms = features[:, 1]
    years = features[:, 0]

    #find_scaling(years, prices)
    #find_scaling(kms, prices)


if __name__ == "__main__2":

    all_data = get_brand_model_entries("toyota", "corolla")
    df = pd.DataFrame(all_data, columns = car_entry_columns)
    df = convert_df_types(df)
    df = get_valid_data(df)

    
    df.to_csv("csv_data/corolla_valid_data.csv")

    df.sort_values(by=["price"], ascending=True, inplace=True)

    #print_df = df[["title", "price", "url"]]
    #urls = df["url"].values
    #print(urls[-1])
    #print(print_df.head(10))

    features = get_year_km_feature(df)
    prices = get_objective(df)

    #print(features)

    df, features, prices = get_all_model_data("toyota", "corolla")

    #print(features)
    #exit()

    #print(len(prices))

    kms = features[:, 1]
    years = features[:, 0]
    scaled_kms = (kms-41)**(0.7)
    scaled_years = (years-2000)**(2.3)

    scaled_features = np.array([scaled_years, scaled_kms])
    scaled_features = scaled_features.transpose()

    #plt.scatter(scaled_kms, prices)
    #plt.show()

    print("-------------year lin regression stats:")
    X_train, X_test, y_train, y_test = train_test_split(scaled_years.reshape(-1, 1), prices, random_state=1)
    reg = LinearRegression().fit(X_train, y_train)
    predictions = reg.predict(X_test)

    full_preds = reg.predict(scaled_years.reshape(-1, 1))
    #find_scaling(full_preds, prices)

    train_predictions = reg.predict(X_train)
    print("train R2: ", reg.score(X_train, y_train))
    print("train Mape: ", mean_absolute_percentage_error(y_train, train_predictions))
    print("test R2: ", reg.score(X_test, y_test))
    print("test Mape: ", mean_absolute_percentage_error(y_test, predictions))

    print("-------------PolyRegr stats")

    X_train, X_test, y_train, y_test = train_test_split(features, prices, random_state=1)
    kms_train = X_train[:, 1]
    years_train = X_train[:, 0]

    poly_regr1 = MultiPolyRegr()
    poly_regr1.fit(years_train.reshape(-1, 1), y_train)

    pred_layer1 = poly_regr1.predict(years_train.reshape(-1, 1))
    poly_regr2 = MultiPolyRegr()

    layer2_features = np.zeros([len(kms_train), 2])
    layer2_features[:, 0] = pred_layer1
    layer2_features[:, 1] = kms_train

    poly_regr2.fit(layer2_features, y_train)

    train_predictions = poly_regr2.predict(layer2_features)

    print("train Mape: ", mean_absolute_percentage_error(y_train, train_predictions))


    print("-------------double lin regression stats")

    X_train, X_test, y_train, y_test = train_test_split(scaled_features, prices, random_state=1)
    layered_lin_regr = LayeredLinRegr()
    layered_lin_regr.fit(X_train, y_train)
    layered_lin_regr.predict_and_score(X_test, y_test)

    print("-------------Layered double polyRegr stats")

    X_train, X_test, y_train, y_test = train_test_split(features, prices, random_state=1)
    layered_regr = LayeredModel([MultiPolyRegr, MultiPolyRegr], [[0], [1], [2, 3, 4, 5]])
    layered_regr.fit(X_train, y_train)
    train_predictions = layered_regr.predict(X_train)
    test_predictions =  layered_regr.predict(X_test)
    print("train MAPE: ", mean_absolute_percentage_error(y_train, train_predictions))
    print("test MAPE: ", mean_absolute_percentage_error(y_test, test_predictions))
    print("train MPE", mean_percentage_error(y_train, train_predictions))
    print("test MPE", mean_percentage_error(y_test, test_predictions))

    full_predictions = layered_regr.predict(features)
    percentage_errors = (prices - full_predictions)/prices

    df["predictions"] = full_predictions
    df["error"] = percentage_errors

    df = df[["title", "submodel", "km", "year", "price", "predictions", "error", "url"]]
    df.to_csv("csv_data/corolla_predictions.csv") 

    larger_predictions = np.where(train_predictions > y_train, 1, 0)

    
    #perc_df = pd.DataFrame(percentage_errors, columns = ["error"])
    #perc_df.to_csv("csv_data/percentage_errors.csv")

    print(f"Number of predictions above real cost: {np.sum(larger_predictions)}, number of entries: {len(train_predictions)}")

if __name__ == "__main__":

    #Tried:
    """
    Kia soul ~70 entries
    Toyota yaris ~120 entries
    ford F150 ~120 valid entries, high quantity of trash
    volkswagen jetta ~120 entries
    corolla 220 entries
    rav4 ~290 entries, polluted by rav4 prime, needs a reparse
    civic ~324 entries
    nissan rogue ~220 entries
    """    
    brand_models = [["toyota", "corolla"], ["toyota", "rav4"], ["honda", "civic"], ["volkswagen", "jetta"], ["nissan", "rogue"], ["ford", "F150"]]

    for brand_model_pair in brand_models:

        all_data = get_brand_model_entries(brand_model_pair[0], brand_model_pair[1])
        df = pd.DataFrame(all_data, columns = car_entry_columns)
        df = convert_df_types(df)
        df = get_valid_data(df)
        print(f"{brand_model_pair[0]} {brand_model_pair[1]}s: {len(df)}")