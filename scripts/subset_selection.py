#!/usr/bin/python3
"""
    subsetSelection performs the subset selection algorithm to determine the
    best selection of attributes
"""

import argparse
import tempfile
import subprocess
import re
import sys

from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing

def check_process(proc):
    """
        given a process check its return code
        and if there is an error we print out
        stdout and stderr
    """
    if proc.wait() != 0:
        lines = proc.stdout.readlines()
        for line in lines:
            print(line.decode(), end="")
        exit()

class RankMethod(object):
    """Holds information for ranking"""
    def __init__(self, location, ranker, metric="NDCG@10", output="ranking.model"):
        """init"""
        self.location = location
        self.ranker = ranker
        self.metric = metric
        self.output = output

    def get_location(self):
        """return output save location"""
        return self.location

    def get_ranker(self):
        """return output save location"""
        return self.ranker

    def get_metric(self):
        """return output save location"""
        return self.metric

    def get_output(self):
        """return output save location"""
        return self.output

    def train(self, data, features=None, model=""):
        """train on data"""
        if model == "":
            model = self.output
        location = "java -jar " + self.location
        ranker = "-ranker " + str(self.ranker)
        train = "-train " + data
        save = "-save " + model
        metric = "-metric2t " + self.metric

        command = location + " " + ranker + " " + train + " " + save + " " + metric

        if features != None:
            command = command + " -feature " + features
        #print("train:", command)
        process = subprocess.Popen(command,
                                   shell=True,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
        check_process(process)

    def score(self, data, features=None, model=""):
        """score using created model"""

        if model == "":
            model = self.output
        location = "java -jar " + self.location
        test = "-test " + data
        metric = "-metric2t " + self.metric
        load = "-load " + model

        command = location + " " + load + " " + test + " " + metric

        if features != None:
            command = command + " -feature " + features

        #print("score:", command)
        process = subprocess.Popen(command,
                                   shell=True,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
        check_process(process)

        tail = subprocess.Popen("tail -n 1",
                                shell=True,
                                stdin=process.stdout,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        check_process(tail)

        result = tail.stdout.readlines()
        ndcg = re.search("0\\.[0-9]*", str(result[0]))
        return float(ndcg.group(0))







class SubsetSelection(object):
    """Performs the subset selection algorithm"""

    def __init__(self, training, testing, ranker, grow=False, backup="ss.restart", log=sys.stdout):
        self.lock = multiprocessing.Lock()
        self.best_attr = -1
        self.best_ndcg = 0
        self.training = training
        self.testing = testing
        self.ranker = ranker
        self.log = log
        try:
            self.restart = open(backup, "x+")
        except FileExistsError:
            self.restart = open(backup, "r+")

        # the current attributes selected
        self.attributes = []
        # the pool of remaining attributes
        self.pool = []

        self.grow = grow

        self.attributes = []
        self.pool = list(range(1, 47))
        if not self.grow:
            self.attributes = list(range(1, 47))

        prev = self.restart.readlines()

        if len(prev) > 0:
            self.log.write("previous found... restarting...\n")
            self.log.flush()
            with open(backup + ".backup", "w+") as backup_file:
                for line in prev:
                    backup_file.write(line)
                    line = line.split(",")[0]
                    if self.grow:
                        self.attributes.append(int(line))
                    else:
                        self.attributes.remove(int(line))
                    self.pool.remove(int(line))

    def ___write_attributes_to_file(self, attr, filename="features.txt"):
        with open(filename, mode="w+") as file:
            file.truncate()
            for i in attr:
                file.write(str(i) + "\n")

    def __done(self):
        if self.grow:
            return len(self.pool) == 0
        return len(self.pool) == 1

    def __run_on_file(self, pair):
        filename = pair["filename"]
        attr = pair["attr"]

        with tempfile.NamedTemporaryFile(mode="w") as model:
            self.ranker.train(self.training, features=filename, model=model.name)
            ndcg = self.ranker.score(self.testing, features=filename, model=model.name)

        self.lock.acquire()
        if ndcg > self.best_ndcg:
            self.best_ndcg = ndcg
            self.best_attr = attr
        self.lock.release()

        return ndcg

    def __generate_candidate_list(self, idx):
        """how to create a new list"""
        candidate = self.attributes[:]
        if self.grow:
            candidate.append(self.pool[idx])
        else:
            candidate.remove(self.pool[idx])
        return candidate

    def run(self):
        """performs subset selection"""
        if len(self.pool) == 0:
            self.log.write("nothing to do...\n")
            self.log.flush()
            return

        msg = "removed"
        if self.grow:
            msg = "added"
        pool = ThreadPool(8)

        # while we can pick an attribute for the next stage
        while not self.__done():

            self.best_attr = -1
            self.best_ndcg = 0

            # generate candidate choices
            candidates = []
            for i in range(0, len(self.pool)):

                attr = self.__generate_candidate_list(i)
                filename = "/tmp/features" + str(self.pool[i]) + ".txt"

                self.___write_attributes_to_file(attr, filename)
                candidates.append({"filename" : filename, "attr" : i})

            pool.map(self.__run_on_file, candidates)

            # log everything
            self.log.write("attribute " + str(self.pool[self.best_attr]) +
                           " " + msg + " and NCDG@10 is now " + str(self.best_ndcg) + "\n")
            self.restart.write(str(self.pool[self.best_attr]) + "," + str(self.best_ndcg) + "\n")
            self.restart.flush()
            self.log.flush()

            # update
            self.attributes = self.__generate_candidate_list(self.best_attr)
            del self.pool[self.best_attr]

        if not self.grow:
            self.log.write("attribute " + str(self.pool[0]) + " removed and NCDG@10 is now 0.0\n")
            self.restart.write(str(self.pool[0]) + ",0.0" + "\n")
            self.restart.flush()
            self.log.flush()

def evaluate(p_results):
    """Given a RankMethod object and a file containing a list of features we exclude some number"""

    ignore = p_results.ignore
    lib = p_results.lib_location
    learner = p_results.learner
    metric = p_results.metric
    output = p_results.output
    log = p_results.log_file
    result = p_results.result
    grow = p_results.grow

    train = p_results.train_file
    test = p_results.test_file
    extra = not p_results.score

    if extra:
        log.write("evaluating using attributes...\n")

    ranker = RankMethod(lib, learner, metric, output)

    with open(result, "r") as file:
        with tempfile.NamedTemporaryFile(mode="w") as tmp_file:
            lines = file.readlines()
            if grow:
                features = list(map(int, lines[:len(lines)-ignore]))
            else:
                features = list(map(int, lines[ignore:]))

            if len(features) == 0:
                log.write("no features, NDCG@10=0.0")
                return
            feature_file = tmp_file.name
            for feature in features:
                tmp_file.write(str(feature) + "\n")
            tmp_file.flush()
            if extra:
                log.write("training...\n")
            ranker.train(train, feature_file, output)
            if extra:
                log.write("scoring...\n")
            ndcg = ranker.score(test, feature_file, output)
            if extra:
                log.write("NDCG@10=" + str(ndcg) + "\n")
            else:
                log.write(str(ndcg) + "\n")



def subset_selection(p_results):
    """performs subset selection"""

    lib = p_results.lib_location
    learner = p_results.learner
    metric = p_results.metric
    output = p_results.output
    log = p_results.log_file
    result = p_results.result
    grow = p_results.grow

    train = p_results.train_file
    test = p_results.test_file

    log.write("performing subset selection...\n")
    log.write("training file: " + train + "\n")
    log.write("test file: " + test + "\n")
    log.write("metric:" + metric + "\n")
    log.write("learner:" + str(learner) + "\n")
    log.write("growing:" + str(grow) + "\n")
    log.flush()

    ranker = RankMethod(lib, learner, metric, output)
    evaluator = SubsetSelection(train, test, ranker, grow, result, log)
    evaluator.run()
    log.flush()

def main():
    """Main method"""
    parser = argparse.ArgumentParser(description="performs subset selection on a set of attributes")
    # evaluate algorithm with subset of attributes

    required = parser.add_argument_group('required arguments')
    required_evaluate = parser.add_argument_group('evaluation arguments')

    # evaluate arguments
    required_evaluate.add_argument("--eval",
                                   action="store_true",
                                   default=False,
                                   dest="evaluate",
                                   help="evaluate results from subset selection")

    required_evaluate.add_argument("--ignore",
                                   action="store",
                                   type=int,
                                   default=-1,
                                   dest="ignore",
                                   help="ignore the first K attributes removed")

    required_evaluate.add_argument("--save",
                                   action="store",
                                   default="output.model",
                                   dest="output",
                                   help="location to save learning model")

    # required
    required.add_argument("--train",
                          action="store",
                          dest="train_file",
                          required=True,
                          help="file used training the model")

    required.add_argument("--test",
                          action="store",
                          dest="test_file",
                          required=True,
                          help="file used for testing the model")

    required.add_argument("--result",
                          action="store",
                          dest="result",
                          required=True,
                          help="results are stored")

    required.add_argument("--lib",
                          action="store",
                          dest="lib_location",
                          required=True,
                          help="location of library")

    required.add_argument("--ranker",
                          action="store",
                          type=int,
                          dest="learner",
                          required=True,
                          help="which ranker to use (refer to RankLib)")
    # optional
    parser.add_argument("--grow",
                        action="store_true",
                        default=False,
                        help="direction to build subsets")

    parser.add_argument("--output",
                        action="store",
                        default=sys.stdout,
                        dest="log_file",
                        help="write output to a file")

    parser.add_argument("--metric",
                        action="store",
                        default="NDCG@10",
                        dest="metric",
                        help="metric used for training")

    parser.add_argument("--score",
                        action="store_true",
                        default=False,
                        help="subpress extraneous output and only report scores")

    results = parser.parse_args()

    if results.log_file != sys.stdout:
        results.log_file = open(results.log_file, "w+")

    if results.evaluate:
        if results.ignore == -1:
            parser.print_help()
            return
        evaluate(results)
    else:
        subset_selection(results)
main()
