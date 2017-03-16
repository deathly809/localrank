#!/usr/bin/python3
"""perform pre-processing"""

import argparse
import sys
import json
import os.path

DEFAULT_FEATURES = range(1, 47)


def extract(line, features):
    """extact the values"""
    result = {}
    for key in features:
        result[key] = line[key]
    return result

def clean(item):
    """clean the item"""

    item = item.split("#")
    comment = None
    if len(item) == 2:
        comment = item[1]

    item = item[0].split()
    rank = item[0]
    qid = item[1]
    result = {}

    item = item[2:]
    for test in item:
        test = test.split(":")
        result[int(test[0])] = test[1]

    return {
        "rank": rank,
        "qid": qid,
        "features": result,
        "comment": comment
    }

def apply_filter(input_file, output_file, features):
    """use list of features to filter"""
    lines = input_file.readlines()
    lines = list(map(clean, lines))

    for i in range(0, len(lines)):
        line = lines[i]
        feat = extract(line["features"], features)
        output_line = line["rank"] + " " + line["qid"]
        for key in features:
            output_line += " " + str(key) + ":" + str(feat[key])
        output_line += " #" + line["comment"]
        output_file.write(output_line)

def load_config(filename="preprocess.conf"):
    """load configuration"""
    features = None
    if os.path.isfile(filename):
        with open(filename) as file:
            result = json.load(file)
            if "features" in result:
                features = result["features"]
            else:
                features = DEFAULT_FEATURES
    else:
        print("missing configuration file... loading default parameters")
        features = DEFAULT_FEATURES
    sorted(features)
    return features


def main():
    """main method"""
    parser = argparse.ArgumentParser(description="performs subset selection on a set of attributes")
    # evaluate algorithm with subset of attributes

    parser.add_argument("--config",
                        action="store",
                        default="preprocess.conf",
                        dest="config_file",
                        help="configuration file")

    parser.add_argument("--input",
                        action="store",
                        required=True,
                        default=sys.stdin,
                        dest="input",
                        help="input file")


    parser.add_argument("--output",
                        action="store",
                        default=sys.stdout,
                        dest="output",
                        help="output location")

    parser.add_argument("--log_file",
                        action="store",
                        default=sys.stdout,
                        dest="log_file",
                        help="write output to a file")


    results = parser.parse_args()

    if results.log_file != sys.stdout:
        results.log_file = open(results.log_file, "w+")
    if results.output != sys.stdout:
        results.output = open(results.output, "w+")
    if results.input != sys.stdin:
        if os.path.isfile(results.input):
            results.input = open(results.input, "r")
        else:
            print("input file does not exist:", results.input)
            exit(1)
    features = load_config(results.config_file)

    apply_filter(results.input, results.output, features)

main()
