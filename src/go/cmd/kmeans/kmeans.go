package main

/* Perform k-means, duh */

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"math"
	"os"
	"strings"

	"sort"

	"github.com/deathly809/research/ranking/src/go/util"
	"github.com/mdesenfants/gokmeans"
)

var (
	input    = flag.String("input", "", "input files to containing DBSCAN results")
	k        = flag.Int("k", 4, "number of clusters")
	iter     = flag.Int("iter", 200, "number of iterations")
	out      = flag.String("out", "", "output file where results shall be saved")
	features = flag.String("features", "", "features to include for clustering")
)

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

func readValues() []gokmeans.Node {
	include := []int(nil)
	if *features != "" {
		f := openFile(*features, false)
		defer f.Close()
		in := bufio.NewScanner(f)
		for in.Scan() {
			v, _ := util.ToInt(in.Text())
			include = append(include, v)
		}
		sort.Ints(include)
	}

	f := openFile(*input, false)
	defer f.Close()

	in := bufio.NewScanner(f)
	in.Split(bufio.ScanLines)

	X := []gokmeans.Node(nil)

	for in.Scan() {
		x := []float64(nil)
		line := in.Text()
		reader := strings.NewReader(line)
		lineScanner := bufio.NewScanner(reader)
		lineScanner.Split(bufio.ScanWords)

		if !lineScanner.Scan() {
			panic("missing class")
		}

		if !lineScanner.Scan() {
			panic("missing qid")
		}

		tmp := include
		for lineScanner.Scan() {
			t := lineScanner.Text()
			if t[0] == '#' {
				break
			}
			id := 0
			value := 0.0
			fmt.Sscanf(t, "%d:%f", &id, &value)

			if include == nil {
				x = append(x, value)
			} else {
				if id == tmp[0] {
					tmp = tmp[1:]
				} else if id > tmp[0] {
					tmp = tmp[1:]
				} else {
					x = append(x, value)
				}
				if len(tmp) == 0 {
					break
				}
			}
		}

		X = append(X, x)
	}
	return X
}

var flags = []string{"input", "out"}

func init() {
	util.ValidateFlags(flags)
}

func main() {
	data := readValues()

	if success, centroids := gokmeans.Train(data, *k, *iter); success {
		oFile := openFile(*out, true)
		defer oFile.Close()

		oFile.WriteString(fmt.Sprintf("%d\n", len(centroids)))

		for _, c := range centroids {
			buff := bytes.NewBuffer(nil)
			for _, v := range c {
				buff.WriteString(fmt.Sprintf("%f ", v))
			}
			buff.WriteRune('\n')
			oFile.WriteString(buff.String())
		}

		clusters := make([][]int, len(centroids))

		for i, x := range data {
			best := 1.0E10
			bIdx := -1

			for j, c := range centroids {
				dist := 0.0
				for i := range c {
					dist += math.Pow(x[i]-c[i], 2.0)
				}

				if dist < best {
					best = dist
					bIdx = j
				}
			}
			clusters[bIdx] = append(clusters[bIdx], i)
		}

		for i, c := range clusters {
			oFile.WriteString(fmt.Sprintf("%d: %d\n", i, len(c)))
			for _, idx := range c {
				oFile.WriteString(fmt.Sprintf("\t%d\n", idx))
			}
		}
	} else {
		fmt.Println("could not create clusters")
	}
}
