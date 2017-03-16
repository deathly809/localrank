package main

/*
 *  Take in results from clustering and original data and partition data into a file for each cluster.
 *
 *	If a cluster has a single classification we do not write it out.
 *
 */

import (
	"bufio"
	"flag"
	"fmt"
	"os"
)

var (
	input     = flag.String("clusterInfo", "", "input files to containing cluster results")
	out       = flag.String("out", "", "output file where results are saved")
	directory = flag.String("dir", "", "location of cluster information")
)

var flags = []string{"clusterInfo", "directory", "out"}

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

func openFile(name string, create bool) *os.File {
	flags := os.O_RDWR
	if create {
		flags |= os.O_CREATE
	}
	f, err := os.OpenFile(name, flags, 0666)

	if err != nil {
		panic(err)
	}
	return f
}

func toInt(input string) int {
	result := 0
	n, err := fmt.Sscanf(input, "%d", &result)
	if n == 0 {
		panic("could not read int")
	}
	if err != nil {
		panic(err)
	}
	return result
}

// Convert to expected format
func main() {
	validateFlags()

	clusterResults := openFile(*input, false)
	defer clusterResults.Close()

	resultReader := bufio.NewScanner(clusterResults)
	resultReader.Split(bufio.ScanLines)

	numClusters := 0

	// NumClusters
	if !resultReader.Scan() {
		panic("missing number of clusters")
	} else {
		numClusters = toInt(resultReader.Text())
	}

	N := 0

	cSize := make([]int, numClusters)

	for i := 0; i < numClusters; i++ {
		if !resultReader.Scan() {
			panic("missing cluster count")
		} else {
			clusterSize := toInt(resultReader.Text())
			cSize[i] = clusterSize
			N += clusterSize
		}
	}

	result := make([]string, N)

	for i := 0; i < numClusters; i++ {
		indexFile := fmt.Sprintf("%s/clustered_index_%d", *directory, i)
		index := openFile(indexFile, false)
		defer index.Close()

		dataFile := fmt.Sprintf("%s/clustered_pred_%d", *directory, i)
		data := openFile(dataFile, false)
		defer data.Close()

		iReader := bufio.NewScanner(index)
		dReader := bufio.NewScanner(data)
		for j := 0; j < cSize[i]; j++ {
			if iReader.Scan() {
				if dReader.Scan() {
					result[toInt(iReader.Text())] = dReader.Text()
				} else {
					panic(fmt.Sprintf("could not read data: index-file=%s data-file=%s\n", indexFile, dataFile))
				}
			} else {
				fmt.Println(indexFile, dataFile)
				panic("could not read index")
			}

		}
	}
	output := openFile(*out, true)
	defer output.Close()

	for _, r := range result {
		output.WriteString(r)
		output.WriteString("\n")
	}

}
