import pandas as pd
import numpy as np
import math
import json


# read in the json files
portfolio = pd.read_json('portfolio.json', orient='records', lines=True)
profile = pd.read_json('profile.json', orient='records', lines=True)
transcript = pd.read_json('transcript.json', orient='records', lines=True)


portfolio.head(10)
portfolio.isnull().sum()

profile.head(10)
profile.isnull().sum()

transcript.head(10)
transcript.isnull().sum()


