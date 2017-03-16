#!/usr/bin/python3
"""filter the trining file using the indices from feature selection"""

import sys
import pandas

def main():
    """main method"""
    if len(sys.argv) != 3:
        print("usage:", sys.argv[0], "data filter")
        exit()
    indices = open(sys.argv[2]).readline().split(",")

    with open("converted_output.txt", "w") as output:
        output.truncate()
        with open(sys.argv[1], "r") as file:
            for line in file:
                # strip off comments
                line = line[:line.find('#') - 1]
                ls = line.split()
                label = ls[0]
                qid = ls[1]
                rem = ls[2:]

                ind = indices

                output_line = label + " " + qid
                for i in range(0, len(rem)):
                    if len(ind) == 0:
                        break
                    if i != ind[0]:
                        output_line += " " + rem[i]
                        ind = ind[1:]
                output.write(output_line)
                output.write("\n")
main()
