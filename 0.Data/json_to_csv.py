# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 18:22:07 2019

@author: Arnold Yu
@discription: This script is modified from yelp data preprocess part. This will convert json file to csv file.
"""

import collections
import csv
import json
from builtins import str

def readAndwrite(input_file, output_file, column_names):
    # Read in the json dataset file and write it out to a csv file, given the column names
    with open(output_file, 'w') as fout:
        csv_file = csv.writer(fout)
        csv_file.writerow(list(column_names))
        with open(input_file, "r", encoding="utf-8") as fin:
            for line in fin:
                line_contents = json.loads(line)
                csv_file.writerow(get_row(line_contents, column_names))

def get_superset_of_column_names_from_file(input_file):
    # Read in the json dataset file and return the superset of column names.
    column_names = set()
    with open(input_file, "r", encoding="utf-8") as fin:
        for line in fin:
            line_contents = json.loads(line)
            column_names.update(
                    set(get_column_names(line_contents).keys())
                    )
    return column_names

def get_column_names(line_contents, parent_key=''):
    """Return a list of flattened key names given a dict.
    Example:
        "hours": {
        "Monday": "10:00-21:00",
        "Tuesday": "10:00-21:00",
        "Friday": "10:00-21:00",
        "Wednesday": "10:00-21:00",
        "Thursday": "10:00-21:00",
        "Sunday": "11:00-18:00",
        "Saturday": "10:00-21:00"
    }
        will return: ['hours.Monday', 'hours.Tuesday','hours.Friday','hours.Wednesday','hours.Thursday','hours.Sunday','hours.Saturday']
    These will be the column names for the eventual csv file.
    """
    column_names = []
    for k, v in line_contents.items():
        column_name = "{0}.{1}".format(parent_key, k) if parent_key else k
        if isinstance(v, collections.MutableMapping):
            column_names.extend(
                    get_column_names(v, column_name).items()
                    )
        else:
            column_names.append((column_name, v))
    return dict(column_names)

def get_nested_value(d, key):
    """Return a list of flattened key names given a dict.
    Example:
        "hours": {
        "Monday": "10:00-21:00",
        "Tuesday": "10:00-21:00",
        "Friday": "10:00-21:00",
        "Wednesday": "10:00-21:00",
        "Thursday": "10:00-21:00",
        "Sunday": "11:00-18:00",
        "Saturday": "10:00-21:00"
    }
         key = 'hours.Monday'
         value = "10:00-21:00"    
    """
    if d == None:
        return None
    if '.' not in key:
        if key in d:
            return d[key]
        return None
    base_key, sub_key = key.split('.', 1)
    if base_key not in d:
        return None
    sub_dict = d[base_key]
    return get_nested_value(sub_dict, sub_key)

def get_row(line_contents, column_names):
    """Return a csv compatible row given column names and a dict."""
    row = []
    for column_name in column_names:
        line_value = get_nested_value(
                        line_contents,
                        column_name,
                        )
        if isinstance(line_value, str):
            row.append('{0}'.format(line_value.encode('utf-8')))
        elif line_value is not None:
            row.append('{0}'.format(line_value))
        else:
            row.append('')
    return row

if __name__ == '__main__':
    """Convert a yelp dataset file from json to csv."""

    column_names = get_superset_of_column_names_from_file('business.json')
    print(column_names)
    readAndwrite('business.json','business.csv', column_names)
    
    
    column_names = get_superset_of_column_names_from_file('review.json')
    print(column_names)
    readAndwrite('review.json','review.csv', column_names)
    
    column_names = get_superset_of_column_names_from_file('user.json')
    print(column_names)
    readAndwrite('user.json','user.csv', column_names)
    """column_names = get_superset_of_column_names_from_file('tip.json')
    print(column_names)
    readAndwrite('tip.json','tip.csv', column_names)
    """
    
    
