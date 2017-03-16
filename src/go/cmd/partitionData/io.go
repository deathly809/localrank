package main

import (
	"bufio"
	"bytes"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/deathly809/research/ranking/src/go/util"
)

func writeClustersToFile(outputName string, cluster1 []util.RankData, cluster2 []util.RankData) {
	buff := bytes.NewBuffer(nil)

	if cluster2 != nil {
		for _, d1 := range cluster1 {
			for _, d2 := range cluster2 {
				buff.WriteString(d1.String())
				buff.WriteString("\n")
				buff.WriteString(d2.String())
				buff.WriteString("\n")
			}
		}
	} else {
		for i1 := 0; i1 < len(cluster1)-1; i1++ {
			for i2 := i1 + 1; i2 < len(cluster1); i2++ {
				buff.WriteString(cluster1[i1].String())
				buff.WriteString("\n")
				buff.WriteString(cluster1[i2].String())
				buff.WriteString("\n")
			}
		}
	}

	writeBufferToFile(outputName, buff)
}

func readFromFile(fName string, indices []mapping) {
	file := util.OpenFile(fName, false)
	defer file.Close()

	v1 := 0.0
	v2 := 0.0

	scan := bufio.NewScanner(file)
	for i := range indices {
		scan.Scan()
		fmt.Sscanf(scan.Text(), "%f", &v1)
		scan.Scan()
		fmt.Sscanf(scan.Text(), "%f", &v2)
		if v1 > v2 {
			indices[i].weight = 1
		} else {
			indices[i].weight = -1
		}
	}
}

// given a filename and a buffer write the contents to a file.
// We create the file if it does not exist and clear it if it
// does
func writeBufferToFile(filename string, buffer *bytes.Buffer) {
	file := util.OpenFile(filename, true)
	defer file.Close()
	file.Write(buffer.Bytes())
}

// write clustering data and meta data to index and data files
func writeClusterData(indexName, dataName string, queries util.RankingData, indices [][]int) {

	dBuffer := bytes.NewBuffer(nil)
	iBuffer := bytes.NewBuffer(nil)

	for _, val := range indices {
		d := queries.QueryData(val[0]).GetDocument(val[1]) //queries[val[0]][val[1]]

		dBuffer.WriteString(d.String())
		dBuffer.WriteString("\n")

		iBuffer.WriteString(fmt.Sprintf("%d\n", d.FilePosition()))
	}

	index := util.OpenFile(indexName, true)
	defer index.Close()
	index.Write(iBuffer.Bytes())

	data := util.OpenFile(indexName, true)
	defer data.Close()
	data.Write(dBuffer.Bytes())
}

func createTempFile(prefix string) string {
	tmpOutput, err := ioutil.TempFile(os.TempDir(), prefix)
	if err != nil {
		panic(err)
	}
	result := tmpOutput.Name()
	tmpOutput.Close()
	return result

}

// save document pairs to a file
func savePairs(fName string, mappings map[int]map[int][]mapping) {
	pairs := util.OpenFile(fName, true)
	defer pairs.Close()

	for _, v1 := range mappings {
		for _, v2 := range v1 {
			for _, d := range v2 {
				if d.weight == 1 {
					pairs.WriteString(fmt.Sprintf("(%d,%d)\n", d.to, d.from))
				} else {
					pairs.WriteString(fmt.Sprintf("(%d,%d)\n", d.from, d.to))
				}
			}
		}
	}
}
