package main

/*
 *  Uses results from createPartion for training
 */

import (
	"bufio"
	"flag"
	"fmt"
	"math"
	"os"
	"runtime"

	"github.com/deathly809/gomath"
	"github.com/gonum/matrix/mat64"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/linear_models"
)

type partitionProperty struct {
	clusterSizes []int
	N            int
}

var (
	directory = flag.String("dir", "", "directory containing values")
	test      = flag.String("test", "", "test data used for training")
	validate  = flag.String("validate", "", "data used to validate")
	nonLinear = flag.Bool("nonlinear", false, "add nonlinear features")
	logistic  = flag.Bool("logistic", false, "combine using logistic regression")
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

var prefixes = []string{"ranknet", "rankboost", "adarank", "lambda"}

func loadXFromFiles(dataType string, N int) [][]float64 {

	result := make([][]float64, N)
	for i := 0; i < N; i++ {
		result[i] = make([]float64, len(prefixes))
	}

	for i, m := range prefixes {
		fileName := fmt.Sprintf("workspace/%s_%s_ranking", m, dataType)
		f := openFile(fileName, false)
		defer f.Close()
		scanner := bufio.NewScanner(f)
		scanner.Split(bufio.ScanLines)
		row := 0
		for scanner.Scan() {
			result[row][i] = toFloat(scanner.Text())
			row++
		}
	}

	if *nonLinear {
		M := len(result[0])
		for i := range result {
			for j := 0; j < M; j++ {
				for k := j; k < M; k++ {
					result[i] = append(result[i], result[i][j]*result[i][k])
				}
			}
		}
	}

	// normalize
	/*
		for i := range result {
			v := 0.0
			for _, x := range result[i] {
				v += x * x
			}
			v = math.Sqrt(v)
			for j := range result[i] {
				result[i][j] /= v
			}
		}
	*/
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
	param := linear_models.NewParameter(linear_models.L2R_LR, C, E)
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
		w[i] /= math.Pow(diff, 1)
		n += w[i]
	}

	for i := range w {
		w[i] /= n
	}
	return w
}

func linearRegression(X [][]float64, Y []float64) ([]float64, float64) {

	rows := len(Y)
	cols := len(X[0])

	fmt.Println(rows, " ", cols)
	observed := mat64.NewDense(rows, 1, nil)
	explVariables := mat64.NewDense(rows, cols, nil)
	qr := new(mat64.QR)
	qr.Factorize(explVariables)
	var q, reg mat64.Dense
	q.QFromQR(qr)
	reg.RFromQR(qr)

	var transposed, qty mat64.Dense
	transposed.Clone(q.T())
	qty.Mul(&transposed, observed)
	regressionCoefficients := make([]float64, cols+1)
	for i := cols - 1; i >= 0; i-- {
		regressionCoefficients[i] = qty.At(i, 0)
		for j := i + 1; j < cols; j++ {
			regressionCoefficients[i] -= regressionCoefficients[j] * reg.At(i, j)
		}
		regressionCoefficients[i] /= reg.At(i, i)
	}

	return regressionCoefficients[1:], regressionCoefficients[0]
}

// Convert to expected format
func main() {

	Y := loadY(*test)
	X := loadXFromFiles("test", len(Y))

	if *logistic {
		for i := range X {
			for _, x := range X[i] {
				fmt.Printf("%7.3f ", x)
			}
			fmt.Println(Y[i])
		}
		//weights, inter := linearRegression(X, Y)
		model := train(X, Y)

		Y = nil
		X = nil
		runtime.GC()

		Y = loadY(*validate)
		X = loadXFromFiles("vali", len(Y))

		f := openFile(fmt.Sprintf("%s/validated", *directory), true)
		defer f.Close()

		for _, x := range X {
			predicted := 2 * linear_models.Predict(model, x)
			/*
				predicted := inter
				for i := range x {
					predicted += x[i] * weights[i]
				}
			*/
			f.WriteString(fmt.Sprintf("%0.16f\n", predicted))
		}
	} else {
		model := trainWeights(X, Y)

		Y = nil
		X = nil
		runtime.GC()

		Y = loadY(*validate)
		X = loadXFromFiles("vali", len(Y))

		f := openFile(fmt.Sprintf("%s/validated", *directory), true)
		defer f.Close()

		fmt.Println(model)

		for _, x := range X {
			predicted := 0.0
			for i := range model {
				predicted += model[i] * x[i]
			}
			f.WriteString(fmt.Sprintf("%0.16f\n", predicted))
		}
	}
}
