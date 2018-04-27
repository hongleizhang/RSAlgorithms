# encoding:utf-8
import sys

sys.path.append("..")
import pandas as pd
from configx.configx import ConfigX

config = ConfigX()
data = pd.read_table(config.trust_path, sep=' ', header=None)
# the number of links
print(len(data))

# the number of followers
print(len(data[0].unique()))

# the number of followees
print(len(data[1].unique()))
