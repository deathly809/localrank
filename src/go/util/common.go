package util

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/deathly809/gomath"
)

const (
	// MissingFlags means that we expected a flag(s) to exist but was not
	// passed in
	MissingFlags = iota
)

// NearestElement reurns the element in list which is closest
func NearestElement(C [][]float64, val []float64, dist Metric) int {
	result := -1
	bDist := 1E100
	for i, c := range C {
		if len(val) != len(c) {
			panic(fmt.Sprintf("%d %d\n", len(val), len(c)))
		}
		nDist := dist(val, c)
		if nDist < bDist {
			result = i
			bDist = nDist
		}
	}
	return result
}

// OpenFile trues to open a file
func OpenFile(name string, create bool) *os.File {
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

// LoadCentroids loads the centroids from the K-means clustering file
func LoadCentroids(filename string) [][]float64 {
	f := OpenFile(filename, false)
	defer f.Close()
	in := bufio.NewScanner(f)

	if !in.Scan() {
		panic("missing number of clusters")
	}
	line := in.Text()
	numClusters := 0
	if n, err := fmt.Sscanf(line, "%d", &numClusters); err != nil {
		panic(err)
	} else if n != 1 {
		panic("missing number of clusters")
	}

	result := make([][]float64, numClusters)
	for i := 0; i < numClusters; i++ {
		if in.Scan() {
			x := []float64(nil)
			line = in.Text()
			toks := bufio.NewScanner(strings.NewReader(line))
			toks.Split(bufio.ScanWords)
			for toks.Scan() {
				v := 0.0
				if n, err := fmt.Sscanf(toks.Text(), "%f", &v); err != nil {
					panic(err)
				} else if n != 1 {
					panic("could not read value")
				}
				x = append(x, v)
			}
			if i > 0 {
				if len(result[i-1]) != len(x) {
					panic("centroid lengths differ")
				}
			}
			result[i] = x
		} else {
			panic("ran out of centroids")
		}
	}
	return result
}

// Metric interace
type Metric func(a, b []float64) float64

// Euclidean distance
func Euclidean(a, b []float64) float64 {
	res := 0.0
	for i := range a {
		t := (a[i] - b[i])
		res += t * t
	}
	return res
}

// Manhattan distance
func Manhattan(a, b []float64) float64 {
	res := 0.0
	for i := range a {
		res += gomath.AbsFloat64(a[i] - b[i])
	}
	return res
}

// Inf difference
func Inf(a, b []float64) float64 {
	res := 0.0
	for i := range a {
		res = gomath.MaxFloat64(res, gomath.AbsFloat64(a[i]-b[i]))
	}
	return res
}

// ToInt is a wrapper for ParseInt
func ToInt(s string) (int, error) {
	t, e := strconv.ParseInt(s, 10, 64)
	return int(t), e
}

// GetMissingFlags returns a list of flags missing from command line
func GetMissingFlags(flags []string) []string {
	res := []string(nil)
	for _, f := range flags {
		if fl := flag.Lookup(f); fl != nil {
			if fl.Value.String() == fl.DefValue {
				res = append(res, f)
			}
		} else {
			res = append(res, f)
		}
	}
	return res
}

// ValidateFlags validates that the flags passed in are present and set
func ValidateFlags(req []string) error {
	flag.Parse()
	if !flag.Parsed() {
		flag.PrintDefaults()
		os.Exit(1)
	}

	missing := GetMissingFlags(req)

	if len(missing) > 0 {
		buf := bytes.NewBuffer(nil)
		buf.WriteString("missing flags: ")
		for _, s := range missing {
			buf.WriteRune(' ')
			buf.WriteString(s)
		}
		fmt.Printf("%s\n", buf.String())
		flag.PrintDefaults()
		os.Exit(MissingFlags)
	}
	return nil
}
