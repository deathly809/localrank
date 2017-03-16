package util

import (
	"bufio"
	"bytes"
	"fmt"
	"strconv"
	"strings"
)

// RankData holds information about ranking data
type RankData struct {
	pos      int
	rank     int
	qid      int
	features []float64
	comment  string
	raw      string
}

// Samples are the set of query document pairs
type Samples map[int][]RankData

// FilePosition returns the line number
func (r RankData) FilePosition() int {
	return r.pos
}

// FeatureCount returns the number of features
func (r RankData) FeatureCount() int {
	return len(r.features)
}

// Rank returns the rank of the document
func (r RankData) Rank() int {
	return r.rank
}

// QID returns the query identifier of the feature vector
func (r RankData) QID() int {
	return r.qid
}

// Comment returns the optional comment at the end of the data
func (r RankData) Comment() string {
	return r.comment
}

// Value returns the ith value
func (r RankData) Value(i int) float64 {
	return r.features[i]
}

// Features returns an array
func (r RankData) Features() []float64 {
	result := make([]float64, len(r.features))
	copy(result, r.features)
	return result
}

// Raw returns the raw input
func (r RankData) Raw() string {
	return r.raw
}

// String returns a string representation of the data
func (r RankData) String() string {
	if r.raw == "" {
		res := bytes.NewBuffer(nil)
		res.WriteString(fmt.Sprintf("%d ", r.rank))
		res.WriteString(fmt.Sprintf("qid:%d ", r.qid))
		for i, v := range r.features {
			res.WriteString(fmt.Sprintf("%d:%f ", i+1, v))
		}
		res.WriteString(r.comment)
		r.raw = res.String()
	}
	return r.raw
}

// Lexer
func readRank(scan *bufio.Scanner) int {
	if !scan.Scan() {
		panic("Could not read rank")
	}

	if v, err := strconv.Atoi(scan.Text()); err != nil {
		panic(err)
	} else {
		return v
	}
}

func readQID(scan *bufio.Scanner) int {
	if !scan.Scan() {
		panic("Could not read qid")
	}

	result := 0
	s := scan.Text()
	if n, err := fmt.Sscanf(s, "qid:%d", &result); err != nil {
		panic(fmt.Sprintf("%s: %s", err.Error(), s))
	} else if n != 1 {
		panic(fmt.Sprintf("could not read qid: %s", s))
	}
	return result
}

func isComment(s string) bool {
	t := []rune(s)
	return t[0] == '#'
}

func readComment(s string, scan *bufio.Scanner) string {
	result := bytes.NewBuffer(nil)
	result.WriteString(s)
	for scan.Scan() {
		result.WriteRune(' ')
		result.WriteString(scan.Text())
	}
	return result.String()
}

type tuple struct {
	id    int
	value float64
}

func readTuple(s string) tuple {
	var result tuple
	if n, err := fmt.Sscanf(s, "%d:%f", &(result.id), &(result.value)); err != nil {
		panic(fmt.Sprintf("%s: %s", err.Error(), s))
	} else if n != 2 {
		panic(fmt.Sprintf("could not read tuple: %s", s))
	}
	return result
}

func parseLine(line string, lineNumber int) RankData {
	var result RankData
	result.pos = lineNumber
	result.raw = line

	if len(line) == 0 {
		return result
	}

	toks := bufio.NewScanner(strings.NewReader(line))
	toks.Split(bufio.ScanWords)

	result.rank = readRank(toks)
	result.qid = readQID(toks)

	x := []float64(nil)
	for toks.Scan() {
		txt := toks.Text()
		if isComment(txt) {
			result.comment = readComment(txt, toks)
			break
		} else {
			x = append(x, readTuple(txt).value)
		}
	}

	result.features = x

	return result
}

// QueryDocumentSet wraps document for a query
type QueryDocumentSet struct {
	documents []RankData
	qid       int
}

func (q *QueryDocumentSet) append(doc RankData) {
	q.documents = append(q.documents, doc)
}

// GetDocument returns the ith document
func (q *QueryDocumentSet) GetDocument(i int) RankData {
	return q.documents[i]
}

// Size returns the number of documents
func (q *QueryDocumentSet) Size() int {
	return len(q.documents)
}

// RankingData wrapper
type RankingData struct {
	data         map[int]*QueryDocumentSet
	featureCount int
	count        int
	ids          []int
}

// FeatureCount returns the number of features in a document-query
func (r RankingData) FeatureCount() int {
	return r.featureCount
}

// QueryCount returns the number of queries
func (r RankingData) QueryCount() int {
	return len(r.data)
}

func (r RankingData) DocumentCount() int {
	return r.count
}

// QueryData returns data for a query id
func (r RankingData) QueryData(qid int) *QueryDocumentSet {
	return r.data[qid]
}

// QueryIds returns the ids for all queries
func (r RankingData) QueryIds() []int {
	if r.ids == nil {
		for qid := range r.data {
			r.ids = append(r.ids, qid)
		}
	}
	return r.ids
}

// LoadRankingData loads Letor ranking data from a file
func LoadRankingData(filename string) RankingData {

	// Open file for reading
	file := OpenFile(filename, false)
	defer file.Close()

	// Scanner goes line by line
	scanner := bufio.NewScanner(file)
	scanner.Split(bufio.ScanLines)

	result := RankingData{}
	result.featureCount = 0
	result.data = make(map[int]*QueryDocumentSet) //[]RankData(nil)

	pos := 0
	for scanner.Scan() {

		d := parseLine(scanner.Text(), pos)

		if result.featureCount == 0 {
			result.featureCount = d.FeatureCount()
		} else {
			if result.featureCount != d.FeatureCount() {
				panic("feature is variadic!")
			}
		}
		pos++

		if result.data[d.qid] == nil {
			result.data[d.qid] = &QueryDocumentSet{nil, d.qid}
		}
		result.data[d.qid].append(d)
	}
	result.count = pos

	return result
}
