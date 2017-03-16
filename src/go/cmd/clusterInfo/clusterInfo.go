package main

/*
 *  Convert the LETOR MQ data sets into the SVMLight format for testing
 */

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/deathly809/research/ranking/src/go/util"
)

var (
	input  = flag.String("input", "", "input files to convert")
	output = flag.String("output", "", "output file for writing")
)

var flags = []string{"input", "output"}

func validateFlags() {
	flag.Parse()
	if !flag.Parsed() {
		flag.PrintDefaults()
		os.Exit(1)
	}

	for _, f := range flags {
		if fl := flag.Lookup(f); fl != nil {
			if fl.Value.String() == fl.DefValue {
				fmt.Fprintf(os.Stderr, "missing %s flag\n", f)
				os.Exit(1)
			}
		}
	}
}

func openFile(name string) *os.File {
	f, err := os.OpenFile(name, os.O_RDWR, 0)

	if err != nil {
		panic(err)
	}
	return f
}

// Convert to expected format
func main() {
	validateFlags()

	fIn := openFile(*input)
	defer fIn.Close()
	input := bufio.NewScanner(fIn)

	fOut := openFile(*output)
	defer fOut.Close()
	output := bufio.NewWriter(fOut)

	input.Split(bufio.ScanLines)

	numClusters := 0

	// NumClusters
	if !input.Scan() {
		panic("missing number of clusters")
	} else {
		txt := input.Text()
		v, _ := util.ToInt(txt)
		numClusters = v
		output.WriteString(txt)
		output.WriteString("\n")
	}

	for i := 0; i < numClusters; i++ {
		if !input.Scan() {
			panic("missing cluster count")
		} else {
			txt := strings.Split(input.Text(), ":")
			txt[0] = strings.TrimSpace(txt[0])
			txt[1] = strings.TrimSpace(txt[1])
			output.WriteString(txt[0])
			output.WriteString(":")
			output.WriteString(txt[1])
			output.WriteString("\n")
			v, _ := util.ToInt(txt[1])
			clusterSize := v
			for j := 0; j < clusterSize; j++ {
				if !input.Scan() {
					panic("Expected cluster element")
				}
			}
		}
	}
	output.Flush()
}
