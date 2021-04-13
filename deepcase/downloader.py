from elasticsearch import Elasticsearch, helpers
from sklearn.preprocessing import LabelEncoder
import argparse
import csv
import json
import numpy as np
import pandas as pd
import urllib3
import warnings

import torch
from preprocessing import Filter

class Downloader(object):

    def __init__(self, url=None, username=None, password=None, verbose=True):
        """Load data from database.

            Parameters
            ----------
            url : string
                URL on which Elasticsearch database is stored.

            username : string
                Username credentials for database

            password : string
                Password credentials for database.
            """
        # Create connection to DB
        if url and username and password:
            self.es = Elasticsearch(url, http_auth=(username, password))
        self.verbose = verbose

    ########################################################################
    #                   General data extraction methods                    #
    ########################################################################

    def load(self, index, max=float('inf')):
        """Scan a given index and yield all items

            Parameters
            ----------
            index : string
                Index in which to search

            max : int, default=float('inf')
                Maximum number of items to extract

            Yields
            ------
            data : dict()
                Data in dictionary format
            """
        # Loop over all items
        for i, d in enumerate(helpers.scan(self.es, index=index, size=10000, raise_on_error=False)):
            if self.verbose: print("Loaded {} items...".format(i+1), end='\r')
            # Stop if we extracted everything
            if i >= max: break
            # Yield the data source
            yield d.get('_source')


    def apply(self, data, key):
        """Apply given key function to each item in data

            Parameters
            ----------
            data : iterable
                Data on which to apply key function

            key : func
                Function to apply to each item in data

            Yields
            ------
            data : Object
                Result from key(datapoint)
            """
        # Loop over all items in data
        for d in data:
            # Extract key
            yield key(d)

    ########################################################################
    #                    Custom data extraction methods                    #
    ########################################################################

    def get_events(
            self,
            keys_extract={'ts_start', 'src_ip', 'source', 'detector_name', 'threat_name', 'operation', 'verification_outcome'},
            keys_encode ={'source', 'detector_name', 'threat_name', 'operation', 'verification_outcome'},
            max=float('inf')
        ):
        """"""
        # Load data
        data = self.load("event-*", max)
        # Extract keys
        data = self.apply(data, key=lambda x: {k: v for k, v in x.items() if k in keys_extract})

        # Get data as dataframe
        df = pd.DataFrame(data)
        # Set index column to time
        df.index = pd.to_datetime(df['ts_start'], unit='ms')
        # Remove timestamps
        del df['ts_start']
        # Sort by time
        df.sort_index(inplace=True)

        # Modify columns
        # Extract source key
        df['source'] = [x.split(':')[0] for x in df['source']]
        # Fill no verification_outcome
        df['verification_outcome'][~df['verification_outcome'].astype(bool)] = 'N/A'

        # Prepare encoding dictionary
        encodings = dict()
        # Encode each key
        for key in keys_encode:
            # Create new label encoder
            le = LabelEncoder()
            # Encode labels
            df[key] = le.fit_transform(df[key])
            # Store encoding
            encodings[key] = le.classes_.tolist()

        # Return data and encodings
        return df, encodings

    def read(self, infile, max=float('inf'), decode=False):
        """"""
        # Initialise encoding
        encoding = {}

        # Read encoding file
        with open("{}.encoding.json".format(infile)) as file:
            # Read encoding as json
            encoding = json.load(file)
            # Transform
            for k, v in encoding.items():
                encoding[k] = {str(i): item for i, item in enumerate(v)}

        # Read input file
        with open(infile) as infile:
            # Create csv reader
            reader = csv.DictReader(infile)

            # Read data
            for i, data in enumerate(reader):
                # Break on max
                if i >= max: break

                # Decode data
                if decode:
                    yield {k: encoding.get(k, {}).get(v, v) for k, v in data.items()}
                # Or yield data
                else:
                    # Yield result as ints where possible
                    result = dict()
                    for k, v in data.items():
                        try:
                            result[k] = int(v)
                        except ValueError:
                            result[k] = v
                    yield result

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser("Elasticsearch Downloader")
    parser.add_argument('--url'     , type=str  , help="URL from which to download", default="https://c01b36e1bb374e3489216c232d93f0cb.us-central1.gcp.cloud.es.io:9243/")
    parser.add_argument('--username', type=str  , help="username credentials"      , default="elastic")
    parser.add_argument('--password', type=str  , help="password credentials"      , default="RlL0Y04LGhgwI0DFa4UXFPDv")
    parser.add_argument('--max'     , type=float, help="maximum number of items to load", default=float('inf'))

    parser.add_argument('--read' , type=str, help="file from which to read")
    parser.add_argument('--write', type=str, help="file from which to write")
    args = parser.parse_args()

    # Create downloader
    downloader = Downloader(args.url, args.username, args.password)

    # Write data
    if args.write:
        # Load data
        df, encoding = downloader.get_events(max=args.max)
        # Write data
        df.to_csv(args.write)
        with open("{}.encoding.json".format(args.write), 'w') as outfile:
            json.dump(encoding, outfile)

    # Read data
    if args.read:

        for data in downloader.read(args.read, decode=False):
            print(data)
