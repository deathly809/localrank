package main

/*
 *  Uses results from createPartion for training
 */

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"runtime"

	"github.com/deathly809/gomath"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/linear_models"
)

type partitionProperty struct {
	clusterSizes []int
	N            int
}

var (
	input  = flag.String("input", "", "input CSV files")
	output = flag.String("output", "", "output file for training information")
	test1  = flag.String("test1", "", "testing data for evaluation of model")
	test2  = flag.String("test2", "", "2nd testing data for evaluation of model")
	test3  = flag.String("test3", "", "3rd testing data for evaluation of model")
	test4  = flag.String("test4", "", "4th testing data for evaluation of model")
	test5  = flag.String("test5", "", "5th testing data for evaluation of model")
)

var flags = []string{"input", "output", "train"}

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
		panic("could not read int")
	}
	if err != nil {
		panic(err)
	}
	return result
}

func readPartition(filename string) partitionProperty {
	input := openFile(filename, false)
	defer input.Close()
	partitionReader := bufio.NewScanner(input)
	partitionReader.Split(bufio.ScanLines)

	// First input is the number of clusters
	if !partitionReader.Scan() {
		fmt.Fprint(os.Stderr, "Could not read number of clusters")
		os.Exit(1)
	}

	result := partitionProperty{}
	numClusters := toInt(partitionReader.Text())

	for i := 0; i < numClusters; i++ {
		if !partitionReader.Scan() {
			fmt.Fprint(os.Stderr, "Could not read cluster size")
			os.Exit(1)
		}
		result.clusterSizes = append(result.clusterSizes, toInt(partitionReader.Text()))
		result.N += result.clusterSizes[i]
	}
	return result
}

func convertInstancesToProblemVec(X base.FixedDataGrid) [][]float64 {
	// Allocate problem array
	_, rows := X.Size()
	problemVec := make([][]float64, rows)

	// Retrieve numeric non-class Attributes
	numericAttrs := base.NonClassFloatAttributes(X)
	numericAttrSpecs := base.ResolveAttributes(X, numericAttrs)

	// Convert each row
	X.MapOverRows(numericAttrSpecs, func(row [][]byte, rowNo int) (bool, error) {
		// Allocate a new row
		probRow := make([]float64, len(numericAttrSpecs))
		// Read out the row
		for i := range numericAttrSpecs {
			probRow[i] = base.UnpackBytesToFloat(row[i])
		}
		// Add the row
		problemVec[rowNo] = probRow
		return true, nil
	})
	return problemVec
}

func convertInstancesToLabelVec(X base.FixedDataGrid) []float64 {
	// Get the class Attributes
	classAttrs := X.AllClassAttributes()
	// Only support 1 class Attribute
	if len(classAttrs) != 1 {
		panic(fmt.Sprintf("%d ClassAttributes (1 expected)", len(classAttrs)))
	}
	// ClassAttribute must be numeric
	if _, ok := classAttrs[0].(*base.FloatAttribute); !ok {
		panic(fmt.Sprintf("%s: ClassAttribute must be a FloatAttribute", classAttrs[0]))
	}
	// Allocate return structure
	_, rows := X.Size()
	labelVec := make([]float64, rows)
	// Resolve class Attribute specification
	classAttrSpecs := base.ResolveAttributes(X, classAttrs)
	X.MapOverRows(classAttrSpecs, func(row [][]byte, rowNo int) (bool, error) {
		labelVec[rowNo] = base.UnpackBytesToFloat(row[0])
		return true, nil
	})
	return labelVec
}

// WrapperValue is an enum
type WrapperValue bool

const (
	_Model    = WrapperValue(true)
	_Constant = WrapperValue(false)
)

// Wrapper just wraps our learned function
type Wrapper struct {
	valueType WrapperValue
	model     *linear_models.LogisticRegression
	class     float64
}

// DataSet holds X and Y values
type DataSet struct {
	raw *base.DenseInstances
	X   [][]float64
	Y   []float64
}

func loadData(partitionInfo partitionProperty) []DataSet {
	dataSets := make([]DataSet, len(partitionInfo.clusterSizes))
	for i := range partitionInfo.clusterSizes {
		data, err := base.ParseCSVToInstances(fmt.Sprintf("%s/part_%d.csv", *output, i), true)

		if err != nil {
			fmt.Fprintf(os.Stderr, "could not parse csv file: %s\n", err.Error())
			os.Exit(1)
		}

		X := convertInstancesToProblemVec(data)
		Y := convertInstancesToLabelVec(data)
		dataSets[i] = DataSet{data, X, Y}
	}
	return dataSets
}

func createModels(data []DataSet) []Wrapper {
	// Model Parameters
	Metric := "l2"
	C := 0.1
	Eps := 0.001

	models := make([]Wrapper, len(data))

	for i, d := range data {
		seen, same := false, true
		prev := -1

		for j := 0; j < len(d.X); j++ {
			d := int(d.Y[j])
			if seen && prev != d {
				same = false
				break
			} else {
				seen = true
				prev = d
			}
		}

		if same {
			models[i] = Wrapper{_Constant, nil, float64(prev)}
		} else {
			if model, err := linear_models.NewLogisticRegression(Metric, C, Eps); err != nil {
				panic(err)
			} else {
				model.Fit(d.raw)
				models[i] = Wrapper{_Model, model, 0.0}
			}
		}
	}

	return models
}

func convertToFormat(models []Wrapper, data DataSet) ([][]float64, []float64) {

	X := make([][]float64, len(data.X))

	// linear
	for _, model := range models {
		pos := 0
		if model.valueType == _Model {
			result, err := model.model.Predict(data.raw)
			if err != nil {
				panic(err)
			}
			for _, y := range convertInstancesToLabelVec(result) {
				X[pos] = append(X[pos], y)
				pos++
			}
		} else {
			for j := 0; j < len(X); j++ {
				X[pos] = append(X[pos], model.class)
				pos++
			}
		}
	}

	// non-linear : poly
	M := len(X[0])
	for i := range X {
		for j := 0; j < M; j++ {
			for k := i; k < M; k++ {
				X[i] = append(X[i], X[i][j]*X[i][k])
			}
		}
	}

	// non-linear : difference
	for i := range X {
		for j := 0; j < M; j++ {
			for k := i + 1; k < M; k++ {
				X[i] = append(X[i], X[i][j]-X[i][k])
			}
		}
	}

	return X, data.Y
}

func runAlgorithm(models []Wrapper, dataSets []DataSet) ([][]float64, []float64) {
	// Return values
	X := [][]float64(nil)
	Y := []float64(nil)

	for _, d := range dataSets {
		x, y := convertToFormat(models, d)
		X = append(X, x...)
		Y = append(Y, y...)
	}
	return X, Y
}

func mergeDataSets(dataSets []DataSet) base.FixedDataGrid {
	N := 0
	for _, d := range dataSets {
		N += len(d.X)
	}

	merged := base.NewDenseInstances()

	if N > 0 {
		specs := []base.AttributeSpec(nil)
		for _, a := range dataSets[0].raw.AllAttributes() {
			specs = append(specs, merged.AddAttribute(a))
		}
		for _, c := range dataSets[0].raw.AllClassAttributes() {
			merged.AddClassAttribute(c)
		}
		merged.Extend(N)

		pos := 0
		for _, data := range dataSets {
			for r := 0; r < len(data.X); r++ {
				for _, a := range specs {
					merged.Set(a, pos, data.raw.Get(a, r))
				}
				pos++
			}
		}
	}
	return merged
}

/* ignore */
var (
	Metric = "l2"
	C      = 0.5
	Eps    = 0.0001
)

func trainMine(models []Wrapper, dataSets []DataSet) *linear_models.Model {
	fmt.Println("training mine...")
	X, Y := runAlgorithm(models, dataSets)
	prob := linear_models.NewProblem(X, Y, 0.0)
	param := linear_models.NewParameter(linear_models.L2R_LR, C, Eps)
	mine := linear_models.Train(prob, param)
	runtime.GC()
	fmt.Println("done training")
	return mine
}

func trainNaive(dataSets []DataSet) *linear_models.LogisticRegression {
	fmt.Println("training naive...")
	merged := mergeDataSets(dataSets)
	runtime.GC()
	naive, err := linear_models.NewLogisticRegression(Metric, C, Eps)
	if err != nil {
		panic(err)
	} else {
		if err = naive.Fit(merged); err != nil {
			panic(err)
		}
	}
	runtime.GC()
	return naive
}

func testMine(testingData *base.DenseInstances, mine *linear_models.Model, models []Wrapper, outputFile string) {
	X, Y := convertToFormat(models, DataSet{testingData, convertInstancesToProblemVec(testingData), convertInstancesToLabelVec(testingData)})

	f := openFile(outputFile, true)
	defer f.Close()

	zeroOff := 0
	oneOff := 0
	twoOff := 0
	err := 0.0
	correct := 0.0
	um := 0
	for i, x := range X {
		y := linear_models.Predict(mine, x)
		if y < 0.5 {
			um++
		}

		if i > 0 {
			f.WriteString(fmt.Sprintf("\n%f", y))
		} else {
			f.WriteString(fmt.Sprintf("%f", y))
		}
		yp := int(y + 0.5)
		e := int(gomath.AbsFloat64(y-Y[i]) + 0.5)
		if e == 0 {
			correct++
			zeroOff++
		} else {
			switch gomath.AbsInt(yp - int(Y[i])) {
			case 1:
				oneOff++
			case 2:
				twoOff++
			}
			err += (y - Y[i]) * (y - Y[i])
		}
	}
	fmt.Println(um, " out of ", len(X))
	N := float64(len(Y))
	fmt.Printf("\tpercent correct=%0.2f%% percent wrong=%0.2f%% correct=%0.0f wrong=%0.0f avg_err=%0.3f\n", 100.0*correct/N, 100.0*(N-correct)/N, correct, N-correct, err/N)
	fmt.Printf("\tzero-off=%d one-off=%d two-off=%d\n", zeroOff, oneOff, twoOff)
}

func testNaive(testingData *base.DenseInstances, naive *linear_models.LogisticRegression) {
	if Ys, err := naive.Predict(testingData); err != nil {
		panic(err)
	} else {
		zeroOff := 0
		oneOff := 0
		twoOff := 0
		err := 0.0
		Y := convertInstancesToLabelVec(testingData)
		YS := convertInstancesToLabelVec(Ys)
		correct := 0.0
		for i, y := range YS {
			e := int(gomath.AbsFloat64(y-Y[i]) + 0.5)
			yp := int(y + 0.5)
			if e == 0 {
				correct++
				zeroOff++
			} else {
				err += (y - Y[i]) * (y - Y[i])
				switch gomath.AbsInt(yp - int(Y[i])) {
				case 1:
					oneOff++
				case 2:
					twoOff++
				}
			}
		}
		N := float64(len(YS))
		fmt.Printf("\tpercent correct=%0.2f%% percent wrong=%0.2f%% correct=%0.0f wrong=%0.0f avg_err=%0.3f\n", 100.0*correct/N, 100.0*(N-correct)/N, correct, N-correct, err/N)
		fmt.Printf("\tzero-off=%d one-off=%d two-off=%d\n", zeroOff, oneOff, twoOff)
	}

}

func runTests(filename, output string, mine *linear_models.Model, naive *linear_models.LogisticRegression, models []Wrapper) {
	runtime.GC()

	fmt.Println("testing over ", filename)

	testingData, err := base.ParseCSVToInstances(filename, true)
	if err != nil {
		panic(err)
	}

	fmt.Println("testing my model...")
	testMine(testingData, mine, models, output)

	fmt.Println()

	fmt.Println("testing naive model...")
	testNaive(testingData, naive)

	fmt.Println()
	fmt.Println()

}

func init() {
	// handle command line
	validateFlags()
	// No output
	base.Silent()
}

func load() ([]Wrapper, []DataSet) {
	fmt.Println("training on ", *input)
	partitionInfo := readPartition(*input)
	dataSets := loadData(partitionInfo)
	models := createModels(dataSets)
	return models, dataSets
}

func train(models []Wrapper, dataSets []DataSet) (*linear_models.Model, *linear_models.LogisticRegression) {
	mine := trainMine(models, dataSets)
	naive := trainNaive(dataSets)
	return mine, naive
}

func testing(mine *linear_models.Model, naive *linear_models.LogisticRegression, models []Wrapper) {

	if *test1 != "" {
		runTests(*test1, fmt.Sprintf("%s/output1", *output), mine, naive, models)
	}

	if *test2 != "" {
		runTests(*test2, fmt.Sprintf("%s/output2", *output), mine, naive, models)
	}

	if *test3 != "" {
		runTests(*test3, fmt.Sprintf("%s/output3", *output), mine, naive, models)
	}

	if *test4 != "" {
		runTests(*test4, fmt.Sprintf("%s/output4", *output), mine, naive, models)
	}

	if *test5 != "" {
		runTests(*test5, fmt.Sprintf("%s/output5", *output), mine, naive, models)
	}
}

// Convert to expected format
func main() {
	models, dataSets := load()
	mine, naive := train(models, dataSets)
	testing(mine, naive, models)
}
