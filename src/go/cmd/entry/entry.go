package main

/*
 *	Check arguments for gpuClustering and then execute
 */

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
)

// Handle Flags

type flagStruct struct {
	flagName  string
	message   string
	errorCode int
}

// Flag Errors
const (
	parseFlagFailure   = 1
	missingInputFlag   = iota
	missingOutputFlag  = iota
	missingCommandFlag = iota
	invalidCommandFlag = iota
	missingDensityFlag = iota
	missingEpsilonFlag = iota
	missingBoundFlag   = iota
)

// Flag error messages
var (
	requiredFlags = []flagStruct{
		flagStruct{
			"input",
			"The input flag is required",
			missingInputFlag,
		},
		flagStruct{
			"output",
			"The output flag is required",
			missingOutputFlag,
		},
		flagStruct{
			"density",
			"density value is required for clustering",
			missingDensityFlag,
		},
		flagStruct{
			"epsilon",
			"epsilon value is required for clustering",
			missingEpsilonFlag,
		},
	}
)

var (
	inputName  = flag.String("input", "", "File in which the data is to be read from")
	outputName = flag.String("output", "", "File in which the data is to be written to")
	command    = flag.String("command", "", "determines if we are clustering or training")
	epsilon    = flag.Float64("epsilon", 0, "epsilon controls how close a point has to be in order to be considered a neighbor")
	density    = flag.Int("density", 0, "density sets the number of points within epislon required to be considered a cluster point")
)

// Given flag information check to validate its existence
// and if it does not exist display and error and exit
func checkFlag(req flagStruct) {
	if fl := flag.Lookup(req.flagName); fl != nil {
		if fl.Value.String() == fl.DefValue {
			fmt.Fprintf(os.Stderr, req.message)
			fmt.Fprintf(os.Stderr, "\n")
			os.Exit(req.errorCode)
		}
	} else {
		panic("Checking for flag that does not exist")
	}
}

func validateFlags() {
	flag.Parse()
	if !flag.Parsed() {
		flag.PrintDefaults()
		os.Exit(parseFlagFailure)
	}

	// handle flags required for everyone
	for _, req := range requiredFlags {
		checkFlag(req)
	}
}

const (
	clusterProgram  = "gpuCluster"
	trainingProgram = "gpuTrain"
)

func main() {
	validateFlags()

	var cmd *exec.Cmd

	switch *command {
	case "cluster":
		cmd = exec.Command(clusterProgram, *inputName, *outputName, fmt.Sprintf("%d", *density), fmt.Sprintf("%f", *epsilon))
	case "train":
		cmd = exec.Command(trainingProgram, *inputName, *outputName, fmt.Sprintf("%d", *density), fmt.Sprintf("%f", *epsilon))
	}

	if err := cmd.Run(); err != nil {
		panic(err)
	}

}
