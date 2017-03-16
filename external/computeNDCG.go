package main

import (
	"bufio"
	"flag"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
)

var (
	input    = flag.String("input", "", "input predictions")
	dataFile = flag.String("data", "", "original data file")
	k        = flag.Int("k", 0, "how many elements to evaluate")
)

var flags = []string{"input", "dataFile", "k"}

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
				flag.PrintDefaults()
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

func parseDataFile() []int {
	dInput := openFile(*dataFile, false)
	sInput := bufio.NewScanner(dInput)
	sInput.Split(bufio.ScanLines)

	qid := -1

	result := []int(nil)

	for sInput.Scan() {
		line := sInput.Text()

		sToken := bufio.NewScanner(strings.NewReader(line))
		sToken.Split(bufio.ScanWords)

		q := 0

		for sToken.Scan() {
			t := sToken.Text()
			if n, err := fmt.Sscanf(t, "qid:%d", &q); n == 1 && err == nil {
				//fmt.Println(q)
			}
		}
		if qid != q {
			result = append(result, 0)
			qid = q
		}
		result[len(result)-1]++
	}
	return result
}

func main() {
	validateFlags()
	querySizes := parseDataFile()
	fmt.Println(len(querySizes), " queries")
	fInput := openFile(*input, false)
	sInput := bufio.NewScanner(fInput)
	sInput.Split(bufio.ScanLines)
	data := make([][]float64, len(querySizes))
	pos := 0
	for sInput.Scan() {

		f, err := strconv.ParseFloat(sInput.Text(), 64)
		if err != nil {
			panic(err)
		}

		data[pos] = append(data[pos], f)
		querySizes[pos]--

		if querySizes[pos] == 0 {
			pos++
		}
	}
	res := 0.0
	total := 0.0
	for _, d := range data {
		for i, v := range d {
			top := math.Pow(2, v) - 1
			bottom := math.Log2(float64(i) + 2)
			res += top / bottom
			total++
			if i == *k {
				break
			}
		}
	}

	fmt.Println(res / total)
}
