import warnings


warnings.simplefilter(action="ignore", category=FutureWarning)



from glob import glob

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from category_encoders import OneHotEncoder
from IPython.display import VimeoVideo
from ipywidgets import Dropdown, FloatSlider, IntSlider, interact
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted


# # Prepare Data

# ## Import

# In[10]:


pd.read_csv("data/mexico-city-real-estate-1.csv").head()


# In[22]:

def wrangle(filepath):
    # Read CSV file
    df = pd.read_csv(filepath)

    # Subset data: Apartments in "Distrito Federal", less than 100,000
    mask_ba = df["place_with_parent_names"].str.contains("Distrito Federal")
    mask_apt = df["property_type"] == "apartment"
    mask_price = df["price_aprox_usd"] < 100_000
    df = df[mask_ba & mask_apt & mask_price]

    # Subset data: Remove outliers for "surface_covered_in_m2"
    low, high = df["surface_covered_in_m2"].quantile([0.1, 0.9])
    mask_area = df["surface_covered_in_m2"].between(low, high)
    df = df[mask_area]

    # Split "lat-lon" column
    df[["lat", "lon"]] = df["lat-lon"].str.split(",", expand=True).astype(float)
    df.drop(columns="lat-lon", inplace=True)

    # Get place name
    df["borough"] = df["place_with_parent_names"].str.split("|", expand=True)[1]
    df.drop(columns="place_with_parent_names", inplace=True)


    # Drop features with high NaN values
    df.drop(columns=["surface_total_in_m2","price_usd_per_m2","floor","rooms","expenses"], inplace=True)

    # Drop low or high cardinality columns
    df.drop(columns=["operation","property_type","currency","properati_url"], inplace=True)

    # Drop leaky columns
    df.drop(columns=['price',
                     'price_aprox_local_currency',
                     'price_per_m2'], inplace=True)
    """"
    # Drop columns with Multicollinearity
    df.drop(columns=["surface_total_in_m2","rooms"], inplace=True)
    """

    return df


# In[23]:


df1 = wrangle("data/mexico-city-real-estate-1.csv")
df1.head()


# In[29]:


df1.info()
#df1.isnull().sum() / len(df1) *100
#df1.select_dtypes("object").nunique()


# In[30]:


files = glob("data/mexico-city-real-estate-*.csv")
files

# In[32]:


frames = [wrangle(file) for file in files]
df = pd.concat(frames, ignore_index=True)
print(df.info())
df.head()

# In[38]:


# Plot distribution of price
plt.hist(df["price_aprox_usd"])
plt.xlabel("Area [sq meters]")
plt.ylabel("Count")
plt.title("Distribution of Apartment Prices")

# In[42]:


# Plot price vs area
plt.scatter(x=df["surface_covered_in_m2"], y=df["price_aprox_usd"])
plt.xlabel("Area [sq meters]")
plt.ylabel("Price [USD]")
plt.title("Mexico City: Price vs. Area");
# Don't delete the code below 👇
plt.savefig("images/2-5-5.png", dpi=150)


# In[50]:


# Split data into feature matrix `X_train` and target vector `y_train`.
features = ["surface_covered_in_m2","lat","lon","borough"]
target = "price_aprox_usd"
X_train = df[features]
y_train = df[target]
X_train.shape


# # Build Model

# In[52]:


y_mean = y_train.mean()
y_pred_baseline = [y_mean] * len(y_train)
baseline_mae = mean_absolute_error(y_train, y_pred_baseline)
print("Mean apt price:", y_mean)
print("Baseline MAE:", baseline_mae)


# In[102]:


# Build Model
model = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    SimpleImputer(),
    Ridge()
)
# Fit model
model.fit(X_train, y_train)


# ## Evaluate

# In[61]:


pd.read_csv("data/mexico-city-test-features.csv").head()


# In[62]:


X_test = pd.read_csv("data/mexico-city-test-features.csv")
print(X_test.info())
X_test.head()


# In[65]:


y_test_pred = pd.Series(model.predict(X_test))
y_test_pred.head()


# In[103]:


coefficients = model.named_steps["ridge"].coef_.round(2)
features = model.named_steps["onehotencoder"].get_feature_names()
feat_imp = pd.Series(coefficients, index=features).sort_values(key=abs)
feat_imp


# In[105]:


feat_imp.sort_values(key=abs).tail(10).plot(
    kind="barh",
    xlabel="Importance [USD]",
    ylabel="Feature",
    title="Feature Importances for Apartment Price");


# In[106]:


# Create horizontal bar chart
feat_imp.sort_values(key=abs).tail(10).plot(
    kind="barh",
    xlabel="Importance [USD]",
    ylabel="Feature",
    title="Feature Importances for Apartment Price");
