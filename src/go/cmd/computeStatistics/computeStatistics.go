package main

import (
	"flag"
	"fmt"

	"sync"

	"github.com/deathly809/research/ranking/src/go/util"
)

var (
	input = flag.String("in", "", "training file")
	csv   = flag.Bool("csv", false, "output in csv format")
)

var flags = []string{"in"}

func init() {
	util.ValidateFlags(flags)
}

func main() {

	data := util.LoadRankingData(*input)

	if data.QueryCount() == 0 {
		fmt.Println("no data in file...")
		fmt.Println(data.FeatureCount())
	} else {
		NumFeatures := data.FeatureCount()

		bucketSize := 30

		min := make([]float64, NumFeatures)
		max := make([]float64, NumFeatures)
		mean := make([]float64, NumFeatures)
		variance := make([]float64, NumFeatures)
		dist := make([][]float64, NumFeatures)
		qMean := make([]float64, NumFeatures)
		qVar := make([]float64, NumFeatures)
		cov := make([][]float64, NumFeatures)
		corr := make([][]float64, NumFeatures)

		for i := range dist {
			dist[i] = make([]float64, bucketSize)
			cov[i] = make([]float64, NumFeatures)
			corr[i] = make([]float64, NumFeatures)
		}

		for i := range min {
			min[i] = 1E100
			max[i] = -1E100
		}

		M := float64(NumFeatures)
		N := 0.0

		lock := &sync.Mutex{}
		wg := &sync.WaitGroup{}

		for _, qID := range data.QueryIds() {
			wg.Add(1)
			qData := data.QueryData(qID)

			go func(d *util.QueryDocumentSet) {

				qM := make([]float64, len(qMean))
				qV := make([]float64, len(qVar))

				for i := 0; i < d.Size(); i++ {
					tmp := d.GetDocument(i)

					features := tmp.Features()
					for i, f := range features {

						lock.Lock()
						if f < min[i] {
							min[i] = f
						}
						if f > max[i] {
							max[i] = f
						}
						dist[i][int(f*(float64(bucketSize)-1E-6))]++
						lock.Unlock()

						qM[i] += f
						qV[i] += f * f
					}
					N++
				}

				for i := range qV {

					lock.Lock()
					mean[i] += qM[i]
					variance[i] += qV[i]

					qMean[i] += qM[i] / M
					qVar[i] += qV[i]/M - (qM[i]*qM[i])/(M*M)
					lock.Unlock()

				}
				wg.Done()
			}(qData)
		}

		wg.Wait()

		same := []int{}

		// Display output
		M = float64(data.QueryCount())
		if *csv == false {
			fmt.Printf("%d data points\n", int(N))
			fmt.Printf("%d queries\n", int(M))
		}
		for i, d := range dist {
			qMean[i] /= M
			qVar[i] /= M

			mean[i] /= N
			variance[i] = variance[i]/N - (mean[i] * mean[i])

			if variance[i] == 0.0 {
				same = append(same, i)
			}
			if *csv {
				for j := 0; j < len(d)-1; j++ {
					fmt.Printf("%0.3f,", d[j]/N)
				}
				if len(d) > 0 {
					fmt.Printf("%0.3f", d[len(d)-1]/N)
				}
			} else {
				fmt.Printf("%3d : \n\tmean=%10f\n\tqMean=%10f\n\tvariance=%8f\n\tqVariance=%8f \n\n\t[ ", i, mean[i], qMean[i], variance[i], qVar[i])
				for _, v := range dist[i] {
					fmt.Printf("%0.3f ", v/N)
				}
				fmt.Println("]")
			}
			fmt.Println()
		}

		if len(same) > 0 {
			fmt.Println("Found zero variance attibutes:")
			for _, s := range same {
				fmt.Print(s, " ")
			}
			fmt.Println()
		}
		/*
			// TODO: Figure out how to do correlation

			for _, d := range data {
				for _, f := range d {
					for i, vi := range f.Features() {
						for j, vj := range f.Features() {
							cov[i][j] += (vi - mean[i]) * (vj - mean[j])
						}
					}
				}
			}

			for i := 0; i < NumFeatures; i++ {
				if variance[i] != 0.0 {
					vi := variance[i]
					for j := 0; j < NumFeatures; j++ {
						cov[i][j] /= N
						cv := cov[i][j]
						vj := variance[j]
						if vj != 0.0 {
							corr[i][j] = (cv / (vj * vi)) / N
						} else {
							corr[i][j] = 0.0
						}
					}
				} else {
					for j := 0; j < NumFeatures; j++ {
						corr[i][j] = 0.0
					}
				}
			}

				for i := 0; i < NumFeatures; i++ {
					for j := 0; j < NumFeatures; j++ {
						fmt.Printf("%0.2E ", corr[i][j])
						if corr[i][j] > 1.001 || corr[i][j] < -1.001 {
							panic("correlation is not between -1 and 1")
						}
					}
					fmt.Println()
				}
		*/

	}
}
