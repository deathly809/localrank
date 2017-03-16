package main

/*
 *  Convert from SVMLight to CSV
 */

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strings"
)

type partitionProperty struct {
	clusterSizes []int
}

var (
	input  = flag.String("input", "", "svmLight input")
	output = flag.String("output", "", "csv output file")
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
				flag.PrintDefaults()
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
		panic(fmt.Sprintf("could not parse to int: %s", input))
	}
	if err != nil {
		panic(err)
	}
	return result
}

func toFloat(input string) float64 {
	result := 0.0
	n, err := fmt.Sscanf(input, "%f", &result)
	if n == 0 {
		panic(fmt.Sprintf("could not parse to float: %s", input))
	}
	if err != nil {
		panic(err)
	}
	return result
}

type pair struct {
	label string
	value string
}

type feature struct {
	classification float64
	attributes     []pair
}

func clean(str string) feature {
	result := feature{}
	line := bufio.NewScanner(strings.NewReader(str))
	line.Split(bufio.ScanWords)

	if !line.Scan() {
		fmt.Fprint(os.Stderr, "could not read classification")
		os.Exit(1)
	}

	// first should be classification
	result.classification = toFloat(line.Text())

	// qid
	if !line.Scan() {
		fmt.Fprint(os.Stderr, "could not read qid")
		os.Exit(1)
	}

	for line.Scan() {
		txt := line.Text()
		if txt[0] == '#' {
			break
		}
		spl := strings.Split(txt, ":")
		if len(spl) != 2 {
			fmt.Fprintf(os.Stderr, "invalid attribute pair: %s", txt)
			os.Exit(1)
		}
		p := pair{spl[0], spl[1]}
		result.attributes = append(result.attributes, p)
	}
	return result
}

// Convert to expected format
func main() {
	validateFlags()

	svmLightFile := openFile(*input, false)
	defer svmLightFile.Close()

	fileReader := bufio.NewScanner(svmLightFile)
	fileReader.Split(bufio.ScanLines)

	attrPositions := make(map[string]int)

	data := []feature(nil)
	largestRecord := 0
	recordID := 0
	for fileReader.Scan() {
		data = append(data, clean(fileReader.Text()))
		for _, a := range data[recordID].attributes {
			if _, ok := attrPositions[a.label]; !ok {
				attrPositions[a.label] = largestRecord
				largestRecord++
			}
		}
		recordID++
	}

	csvFile := openFile(*output, true)
	defer csvFile.Close()

	fileWriter := bufio.NewWriter(csvFile)
	defer fileWriter.Flush()

	line := make([]string, largestRecord)

	// Write header
	for k, v := range attrPositions {
		line[v] = k
	}

	for i := 0; i < len(line); i++ {
		fileWriter.WriteString(line[i])
		fileWriter.WriteRune(',')
	}
	fileWriter.WriteString("class\n")

	// Write data
	for _, d := range data {
		for i := range line {
			line[i] = ""
		}
		for _, a := range d.attributes {
			line[attrPositions[a.label]] = a.value
		}

		for _, v := range line {
			fileWriter.WriteString(v)
			fileWriter.WriteRune(',')
		}
		fileWriter.WriteString(fmt.Sprintf("%f\n", d.classification))
	}

}
