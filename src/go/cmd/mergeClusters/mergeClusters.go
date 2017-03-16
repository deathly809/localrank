package main

/*
 *  Use cluster information to place new values in their appropriate cluster
 *
 *	Create three file types:
 *		clustered_meta
 *		clustered_index_[0-9]+
 *		clustered_data_[0-9]+
 *
 *		clustered_meta holds information about the partitions
 *		each clustered_data_i has the data that is closest to cluster i
 *		each clustered_index_i has the original position of the data
 *
 */

import (
	"flag"
	"sort"

	"github.com/deathly809/research/ranking/src/go/util"
)

var (
	input1 = flag.String("in1", "", "first input file")
	input2 = flag.String("in2", "", "second input file")
	output = flag.String("out", "", "output file")
)

var flags = []string{"in1", "in2", "out"}

// ByPos is derp
type ByPos []util.RankData

func init() {
	util.ValidateFlags(flags)
}

func (p ByPos) Len() int           { return len(p) }
func (p ByPos) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }
func (p ByPos) Less(i, j int) bool { return p[i].Rank() > p[j].Rank() }

// Convert to expected format
func main() {
	first := util.LoadRankingData(*input1)
	second := util.LoadRankingData(*input2)

	N := first.DocumentCount() + second.DocumentCount()

	result := make(map[int][]util.RankData)

	for _, qID := range first.QueryIds() {
		qData := first.QueryData(qID)
		for i := 0; i < qData.Size(); i++ {
			result[qID] = append(result[qID], qData.GetDocument(i))
		}
	}

	for _, qID := range second.QueryIds() {
		qData := second.QueryData(qID)
		for i := 0; i < qData.Size(); i++ {
			result[qID] = append(result[qID], qData.GetDocument(i))
		}
		sort.Sort(ByPos(result[qID]))
	}

	output := util.OpenFile(*output, true)
	defer output.Close()

	for _, v := range result {
		for _, d := range v {
			N--
			output.WriteString(d.Raw())
			output.WriteString("\n")
		}
	}
	if N != 0 {
		panic("did not output everything!")
	}
}
