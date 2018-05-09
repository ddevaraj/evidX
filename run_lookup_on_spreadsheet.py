from __future__ import print_function, division
import pandas as pd
import datetime
import argparse

if __name__ == '__main__':

    start = datetime.datetime.now()

    parser = argparse.ArgumentParser()

    parser.add_argument('inputFile', help='Input file')
    parser.add_argument('inputColumn', help='Input file')
    parser.add_argument('lookupFile', help='Lookup file')
    parser.add_argument('lookupTarget', help='Lookup Column')

    args = parser.parse_args()

    df1 = pd.read_csv(args.lookupFile, sep='\t', header=0, index_col=0)

    df = pd.read_csv(args.inputFile, sep='\t', header=0)

    df[args.lookupTarget] = df1.loc[df[args.inputColumn]][args.lookupTarget].tolist()

    df.to_csv(args.inputFile, sep="\t")
