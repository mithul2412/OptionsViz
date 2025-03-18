import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from copy import copy
from PIL import Image
import requests
from io import BytesIO

def add_nums(x, y):
    return x+y