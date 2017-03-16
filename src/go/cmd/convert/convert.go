package main

/*
 *  Convert the LETOR MQ data sets into the SVMLight format for testing
 */

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"os"
	"strings"
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

func parseLine(in string) string {
	buffer := bytes.NewBuffer(nil)
	tok := bufio.NewScanner(strings.NewReader(in))
	tok.Split(bufio.ScanWords)

	if !tok.Scan() {
		panic("missing class")
	}

	class := 0
	fmt.Sscanf(tok.Text(), "%d", &class)
	buffer.WriteString(fmt.Sprintf("%d", class))

	for tok.Scan() {
		txt := tok.Text()
		if txt[0] == '#' {
			break
		}
		buffer.WriteString(" ")
		buffer.WriteString(txt)
	}
	return buffer.String()
}

// Convert to expected format
func main() {
	validateFlags()

	buffer := bytes.NewBuffer(nil)

	// Read everything in
	func(b *bytes.Buffer) {
		iFile := openFile(*input, false)
		defer iFile.Close()

		input := bufio.NewScanner(iFile)

		for input.Scan() {
			buffer.WriteString(parseLine(input.Text()))
			buffer.WriteString("\n")
		}
	}(buffer)

	oFile := openFile(*output, true)
	defer oFile.Close()

	output := bufio.NewWriter(oFile)
	output.WriteString(buffer.String())
	output.Flush()
}
