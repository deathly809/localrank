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
	models    = flag.Int("models", -1, "how many models generated")
	directory = flag.String("dir", "", "directory containing values")
	test      = flag.String("test", "", "test data used for training")
	validate  = flag.String("validate", "", "data used to validate")
)

var flags = []string{"models", "directory", "validate", "test"}

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

func init() {
	validateFlags()
	base.Silent()
}

func loadY(filename string) []float64 {
	// Open file for reading
	file := openFile(filename, false)
	defer file.Close()

	// Scanner goes line by line
	scanner := bufio.NewScanner(file)
	scanner.Split(bufio.ScanLines)

	result := []float64(nil)

	for scanner.Scan() {
		line := scanner.Text()
		y := 0.0
		if n, err := fmt.Sscanf(line, "%f ", &y); err != nil {
			panic(err)
		} else if n != 1 {
			panic("could not read class")
		}
		result = append(result, y)
	}

	return result
}

func loadYFromFiles(prefix string) []float64 {
	result := []float64(nil)
	for i := 0; i < *models; i++ {
		f := openFile(fmt.Sprintf("%s%d", prefix, i), false)
		defer f.Close()
		scanner := bufio.NewScanner(f)
		scanner.Split(bufio.ScanLines)

		for scanner.Scan() {
			v := 0.0
			fmt.Sscanf(scanner.Text(), "%f ", &v)
			result = append(result, v)
		}
	}
	return result
}

func loadXFromFiles(prefix string, N int) [][]float64 {

	result := make([][]float64, N)
	for i := 0; i < N; i++ {
		result[i] = make([]float64, *models)
	}

	for i := 0; i < *models; i++ {
		f := openFile(fmt.Sprintf("%s%d", prefix, i), false)
		defer f.Close()
		scanner := bufio.NewScanner(f)
		scanner.Split(bufio.ScanLines)
		row := 0
		for scanner.Scan() {
			result[row][i] = toFloat(scanner.Text())
			row++
		}
	}

	M := len(result[0])
	for i := range result {
		for j := 0; j < M; j++ {
			for k := j; k < M; k++ {
				result[i] = append(result[i], result[i][j]*result[i][k])
			}
		}
	}
	return result
}

func toFloat(s string) float64 {
	result := 0.0
	n, err := fmt.Sscanf(s, "%f", &result)
	if err != nil {
		panic(err)
	}
	if n != 1 {
		panic(fmt.Sprint("could not parse value to float: ", s))
	}

	return result
}

func train(X [][]float64, Y []float64) *linear_models.Model {
	C := 5.0
	E := 0.001
	prob := linear_models.NewProblem(X, Y, 0.0)
	param := linear_models.NewParameter(linear_models.L2R_LR_DUAL, C, E)
	return linear_models.Train(prob, param)
}

func trainWeights(X [][]float64, Y []float64) []float64 {

	if len(Y) == 0 {
		panic("No data points")
	}

	if len(X[0]) == 0 {
		panic("No attributes")
	}

	w := make([]float64, len(X[0]))

	for i := range X {
		for j := range w {
			w[j] += X[i][j]
		}
	}

	n := 0.0
	for i := range w {
		w[i] /= float64(len(Y))
		diff := gomath.AbsFloat64(w[i]-Y[i]) + 0.01
		w[i] /= diff
		n += w[i]
	}

	for i := range w {
		w[i] /= n
	}
	return w
}

// Convert to expected format
func main() {

	fmt.Printf("num_models=%d save_directory=%s validate_file=%s\n", *models, *directory, *validate)

	Y := loadY(*test)
	X := loadXFromFiles(fmt.Sprintf("%s/pred_", *directory), len(Y))

	model := trainWeights(X, Y)

	Y = nil
	X = nil
	runtime.GC()

	Y = loadY(*validate)
	X = loadXFromFiles(fmt.Sprintf("%s/vali_", *directory), len(Y))

	f := openFile(fmt.Sprintf("%s/validated", *directory), true)
	defer f.Close()

	for _, x := range X {
		//predicted := linear_models.Predict(model, x)
		predicted := 0.0
		for i := range model {
			predicted += model[i] * x[i]
		}
		f.WriteString(fmt.Sprintf("%f\n", predicted))
	}
}
