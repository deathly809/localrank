package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
)

var (
	input  = flag.String("input", "", "input values to be converted")
	output = flag.String("output", "", "converted values saved here")
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
				flag.PrintDefaults()
				os.Exit(1)
			}
		}
	}
}

func init() {
	validateFlags()
}
func openFile(name string, create bool) *os.File {
	flags := os.O_RDWR
	if create {
		flags |= os.O_CREATE | os.O_TRUNC
	}
	f, err := os.OpenFile(name, flags, 0666)

	if err != nil {
		panic(err)
	}
	return f
}

type data struct {
	to, weight int
}

func readInput(fName string) map[int][]int {
	in := openFile(fName, false)
	defer in.Close()
	result := make(map[int][]int)
	scan := bufio.NewScanner(in)
	for scan.Scan() {
		var from, to int
		fmt.Sscanf(scan.Text(), "(%d,%d)\n", &from, &to)
		result[from] = append(result[from], to)
	}
	return result
}

func main() {
	data := readInput(*input)

	out := openFile(*output, true)
	defer out.Close()

	out.WriteString("Source,Target,Type,weight\n")
	for k, v := range data {
		for _, d := range v {
			out.WriteString(fmt.Sprintf("%d,%d,Directed,1.0\n", k, d))
		}
	}

}
