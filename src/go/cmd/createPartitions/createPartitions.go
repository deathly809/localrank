package main

/*
 *  Take in results from clustering and original data and partition data into a file for each cluster.
 *
 *	If a cluster has a single classification we do not write it out.
 *
 */

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"strings"

	"github.com/deathly809/research/ranking/src/go/util"
)

var (
	min    = flag.Int("min", 0, "minimum size of a cluster")
	input  = flag.String("input", "", "input files to containing cluster results")
	data   = flag.String("data", "", "input file containing data use from cluster results")
	outDir = flag.String("out", "", "output directory where results shall be saved")
)

var requiredFlags = []string{"input", "data", "out"}

func init() {
	util.ValidateFlags(requiredFlags)
}

func toInt(input string) int {
	if res, err := util.ToInt(input); err != nil {
		panic(err)
	} else {
		return res
	}
}

func readInt(in *bufio.Scanner) int {
	if in.Scan() {
		return toInt(strings.TrimSpace(in.Text()))
	}
	panic("could not read integer")
}

func readClusterData() [][]int {
	clusterResults := util.OpenFile(*input, false)
	defer clusterResults.Close()

	resultReader := bufio.NewScanner(clusterResults)
	resultReader.Split(bufio.ScanLines)

	numClusters := readInt(resultReader)

	// Read centroids
	for i := 0; i < numClusters; i++ {
		resultReader.Scan()
	}

	clusterInfo := make([][]int, numClusters)

	for i := 0; i < numClusters; i++ {
		if !resultReader.Scan() {
			panic("missing cluster count")
		} else {
			clusterSize, id := 0, 0
			line := resultReader.Text()
			if n, e := fmt.Sscanf(line, "%d: %d", &id, &clusterSize); e != nil {
				panic(e)
			} else if n != 2 {
				panic(fmt.Sprintf("expected 'id: count' format, found '%s'\n", line))
			} else {
				for j := 0; j < clusterSize; j++ {
					clusterInfo[i] = append(clusterInfo[i], readInt(resultReader))
				}
			}
		}
	}
	return clusterInfo
}

func readDataFile() []string {
	dataInputFile := util.OpenFile(*data, false)
	defer dataInputFile.Close()
	dataReader := bufio.NewScanner(dataInputFile)

	data := []string(nil)

	for dataReader.Scan() {
		data = append(data, dataReader.Text())
	}
	return data
}

func writePartition(filename string, lines []string) {
	output := util.OpenFile(filename, true)
	defer output.Close()
	for _, l := range lines {
		output.WriteString(l)
		output.WriteString("\n")
	}
}

func filterLines(filter []int, lines []string) []string {
	result := []string(nil)
	for _, idx := range filter {
		result = append(result, lines[idx])
	}
	return result
}

func writePartitions(clusterInfo [][]int) {
	data := readDataFile()
	for i, cluster := range clusterInfo {
		if len(cluster) >= *min {
			filtered := filterLines(cluster, data)
			writePartition(fmt.Sprintf("%s/part_%d", *outDir, i), filtered)
		}
	}
}

func writeReport(clusterInfo [][]int) {
	outputReport := util.OpenFile(fmt.Sprintf("%s/part_report", *outDir), true)
	defer outputReport.Close()

	report := bytes.NewBuffer(nil)
	for i, c := range clusterInfo {
		report.WriteString(fmt.Sprintf("%d:%d\n", len(c), i))
	}
	outputReport.WriteString(fmt.Sprintf("%d\n", len(clusterInfo)))
	outputReport.Write(report.Bytes())
}

// Convert to expected format
func main() {
	clusterInfo := readClusterData()
	writePartitions(clusterInfo)
	writeReport(clusterInfo)
}
