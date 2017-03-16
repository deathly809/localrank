package main

/*
 *	Go over each result and only report those that
 *  have enough and not too many clusters
 */

import (
	"bufio"
	"flag"
	"fmt"
	"os"
)

var (
	directory     = flag.String("input", "", "direcory containing files to parse")
	outputName    = flag.String("output", "", "output file for writing")
	minNumCluster = flag.Int("min", -1, "minimum number of clusters allowed")
	maxNumCluster = flag.Int("max", -1, "maximum number of clusters allowed")
	startK        = flag.Int("kStart", -1, "initial k")
	stepK         = flag.Int("kStep", -1, "k step size")
	endK          = flag.Int("kEnd", -1, "k upper bound")
	startE        = flag.Float64("epsStart", -1, "initial epsilon")
	stepE         = flag.Float64("epsStep", -1, "epsilon step size")
	endE          = flag.Float64("epsEnd", -1, "epsilon upper bound")
)

var flags = [10]string{"input", "output", "min", "max", "kStart", "kStep", "kEnd", "epsStart", "epsStep", "epsEnd"}

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

func countClusters(file string) int {
	result := 0
	input := bufio.NewScanner(openFile(file))

	if !input.Scan() {
		panic("empty file")
	}

	n, err := fmt.Sscanf(input.Text(), "%d", &result)
	if n != 1 {
		panic("could not read cluster count")
	}
	if err != nil {
		panic(err)
	}
	return result

}

func trimZeros(in string) string {
	start := 0
	end := len(in) - 1
	for start < end {
		if in[end] == '0' {
			end--
		} else {
			break
		}
	}
	return in[0 : end+1]
}

// ResultInfo is the best info
type ResultInfo struct {
	name  string
	lines int
	K     int
	E     float64
}

func main() {
	validateFlags()
	f := openFile(*outputName)
	defer f.Close()

	output := bufio.NewWriter(f)
	defer output.Flush()

	var results []ResultInfo

	defer func() {
		if r := recover(); r != nil {
		}
		output.WriteString(fmt.Sprintf("%d\n", len(results)))
		for _, c := range results {
			str := fmt.Sprintf("%s: Clusters=%d K=%d Epsilon=%f\n", c.name, c.lines, c.K, c.E)
			fmt.Fprint(output, str)
		}
	}()
	sVal := flag.Lookup("epsStep").Value.String()
	width := len(sVal) - 2
	fmtString := fmt.Sprintf("%%s/%%d_%%0.%df.txt", width)
	//fmt.Printf("kStep=%s, width=%d, fmt=%s\n", sVal, width, fmtString)

	for K := *startK; K <= *endK; K += *stepK {
		for E := *startE; E <= *endE; E += *stepE {
			filename := fmt.Sprintf(fmtString, *directory, K, E)
			lines := countClusters(filename)

			if lines <= *maxNumCluster && lines >= *minNumCluster {
				results = append(results, ResultInfo{filename, lines, K, E})
			}
		}
	}

}
