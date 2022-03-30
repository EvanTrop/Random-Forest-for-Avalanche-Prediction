//Evan Trop, etrop
//Final Project : Predicting Avalanche Problems with the Random Forest algorithm
//December 11,2020
package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"time"

	. "github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"
)

func main() {

	// //Get directory and target var from command line
	dir := os.Args[1]
	target := os.Args[2]
	//
	// //Set file name for train and test data
	trainFile := filepath.Join(dir, "goTrain.csv")
	testFile := filepath.Join(dir, "goTest.csv")
	//
	// //Create reader object from csv files
	reader1, err1 := os.Open(trainFile)
	if err1 != nil {
		fmt.Println("Error: couldn't open the file")
		os.Exit(1)
	}
	defer reader1.Close()

	reader2, err2 := os.Open(testFile)
	if err2 != nil {
		fmt.Println("Error: couldn't open the file")
		os.Exit(1)
	}
	defer reader2.Close()
	//
	// //Read from csv into dataframe, and set target variable
	trainData := ReadCSV(reader1)
	testData := ReadCSV(reader2)

	if target == "p" {
		target = "o_PersistentSlab_Likelihood"
	} else if target == "w" {
		target = "o_WindSlab_Likelihood"
	} else {
		panic("Error with target arguement. Should be either p or w")
	}

	//Set seed for random num gen
	rand.Seed(time.Now().UnixNano())

	//Build random forest model
	forest := BuildRandomForest(target, int(math.Sqrt(float64(trainData.Ncol()-17))), trainData.Nrow(), 3, trainData)
	predictions := make([]string, testData.Nrow())

	//
	for i := 0; i < testData.Nrow(); i++ {
		row := testData.Subset([]int{i})
		predictions[i] = forest.PredictRow(row)
	}
	//print confusion matrix
	ConfusionMatrix(predictions, target, testData)
}

// Input: integer values for the number of random samples to draw from the data for each trees,
// number of trees to create,and the Data as a dataframe type
// Output: a slice of numTrees Tree objects
func BuildRandomForest(target string, numRandFeatures, numRandSamples, numTrees int, Data DataFrame) randForest {
	forest := make(randForest, numTrees)

	for i := 0; i < numTrees; i++ {
		randData := GetRandomSamples(numRandSamples, Data)
		randFeatures := GetRandomFeatures(numRandFeatures, randData)

		//Hard coding the values for tree attributes, create decision tree object, and create root node
		var decisionTree Tree
		decisionTree.minNumLeaves = 5
		decisionTree.maxDepth = 10
		decisionTree.minSamplesSplit = 15

		//Create root node and set its attributes
		var root Node
		root.InitializeNode(0, target, Data)

		//Set the root node in the decision tree, build the tree from the root,
		//and save the decision tree in the forest
		decisionTree.Root = &root
		decisionTree.BuildTree(&root, target, randFeatures, randData)
		forest[i] = &decisionTree
	}

	return forest
}

// // Input: integer values for the minimum number of samples at an internal node required to split the node,
// // and a randomly generated dataframe from the original data
// // Output: nothing just builds the tree
func (tree Tree) BuildTree(root *Node, target string, randFeatures []string, randData DataFrame) {

	//Base case
	if root.level == tree.maxDepth || root.numSamples == tree.minNumLeaves || root.giniIndex == 0.00 {

		root.giniIndex = GiniIndex(target, randData)

		//Recursive Call
	} else {
		condition, gini := FindBestSplit(target, randData, randFeatures)
		root.conditon = condition
		root.giniIndex = gini
		tData, fData := SplitData(randData, condition)

		//Updates randFeatures since randFeatures is a slice
		//RemoveFeature(condition.col, randFeatures)

		//Create two daughter nodes, initialize attributes for daughters
		var tChild, fChild Node
		tChild.InitializeNode(root.level+1, target, tData)
		fChild.InitializeNode(root.level+1, target, fData)

		//Set the parent for each daughter
		root.trueChild = &tChild
		root.falseChild = &fChild

		//Recursive call on both daughter nodes
		tree.BuildTree(&tChild, target, randFeatures, tData)
		tree.BuildTree(&fChild, target, randFeatures, fData)
	}

}

//Input: A feature to remove from a list of features
//Output: updates the features slice
func RemoveFeature(feature string, features []string) {
	index := 0

	for i, label := range features {
		if label == feature {
			index = i
			break
		}
	}
	features = append(features[:index], features[index+1:]...)
}

//Input: number of random samples to take from the input Data and the data as a DataFrame
//Output: a dataframe compiled of random samples
//This is considered "Baggining" of the data
func GetRandomSamples(numSamples int, Data DataFrame) DataFrame {
	totalRows := Data.Nrow()
	indexes := make([]int, numSamples)

	for i := 0; i < numSamples; i++ {
		indexes[i] = int(rand.Float64() * float64(totalRows))
	}
	randData := Data.Subset(indexes)

	return randData
}

//Input: the number of features to randomly select as a subset from all of the numFeatures
//Output: a list of the indexes for the randomly selected features
func GetRandomFeatures(numFeatures int, randomData DataFrame) []string {
	subsetIndexes := make([]string, numFeatures)
	colLabels := randomData.Names()

	for i := 0; i < numFeatures; i++ {
		index := int(math.Round(rand.Float64() * float64(len(colLabels))))
		subsetIndexes[i] = colLabels[index]
	}

	return subsetIndexes
}

//Input: the target variable's label and some set of data as a dataframe
//Output: the gini index based on the target variable's categories
func GiniIndex(target string, data DataFrame) float64 {
	giniIndex := 1.0

	catCounts := CategoryCounts(target, data)
	numSamples := data.Nrow()

	for _, count := range catCounts {
		Prob_cat := float64(count) / float64(numSamples)
		giniIndex -= Prob_cat * Prob_cat
	}
	return giniIndex
}

func FindBestSplit(target string, data DataFrame, features []string) (Condition, float64) {

	var bestCondition Condition
	minGini := 1.00 ////CHANGED THIS BECAUSE I HAD IT AS ZERO INCORRECTLY
	numSamples := float64(data.Nrow())

	for _, feature := range features {
		uniqueVals := UniqueColVals(data.Col(feature))

		for val, _ := range uniqueVals {
			testCond := Condition{feature, val}
			trueData, falseData := SplitData(data, testCond)

			trueGini := GiniIndex(target, trueData)
			falseGini := GiniIndex(target, falseData)

			tSize, fSize := float64(trueData.Nrow()), float64(falseData.Nrow())

			weightedGini := trueGini*(tSize/numSamples) + falseGini*(fSize/numSamples)

			if weightedGini < minGini {
				minGini = weightedGini
				bestCondition.col = testCond.col
				bestCondition.val = testCond.val
			}

		}
	}
	fmt.Println(minGini)
	return bestCondition, minGini
}

//Input: the data to be split as a DataFrame and the condition to split upon
//Output: two datasets as Dataframe objects split on the data
func SplitData(data DataFrame, condition Condition) (DataFrame, DataFrame) {
	// leftData := New()
	// rightData := New()

	trueRows := make([]int, 0)
	falseRows := make([]int, 0)

	colOfInterest := data.Col(condition.col)

	for i := 0; i < data.Nrow(); i++ {
		rowVal := colOfInterest.Elem(i).Float()

		if Compare(rowVal, condition) {
			trueRows = append(trueRows, i)
		} else {
			falseRows = append(falseRows, i)
		}
	}

	trueData := data.Subset(trueRows)
	falseData := data.Subset(falseRows)

	return trueData, falseData
}

//Input: the target variable's label as a string and the data as a DataFrame
//Output: a map corresponding to the counts for each category of the target variable
func CategoryCounts(target string, data DataFrame) map[string]int {
	//There are 5 categories for avalanche problems 0(unlikely) to certain (4)
	catCounts := make(map[string]int, 5)
	targetCol := data.Col(target)

	for i := 0; i < data.Nrow(); i++ {
		category := targetCol.Elem(i).String()
		catCounts[category] += 1
	}
	return catCounts
}

//Input: takes in a feature column as a Series object
//Output: returns a map with float64s as keys corresponding to the unique values in the list and
//index as the value for the key
func UniqueColVals(column series.Series) map[float64]int {
	uniqueVals := make(map[float64]int, column.Len())

	for i := 0; i < column.Len(); i++ {

		val := column.Elem(i).Float()
		if _, found := uniqueVals[val]; found != true {
			uniqueVals[val] = i
		}
	}
	return uniqueVals
}

//Input: a rowvalue from the dataset as a float64  and a condition object
//Output: a boolean whether the row has
func Compare(rowVal float64, condition Condition) bool {

	if rowVal >= condition.val {
		return true
	}
	return false
}

//Input: A map with key values as the categories of the target variable and
//the value in the map as the count for each category in the dataset
//Output: The majority vote category or if there is a tie then randomly select
//a winning category
func ReturnCategory(counts map[string]int) string {
	keys := make([]string, 0)
	max := 0

	for _, val := range counts {
		if val > max {
			max = val
		}
	}

	for key, val := range counts {
		if val == max {
			keys = append(keys, key)
		}
	}

	if len(keys) > 1 {
		index := rand.Intn(len(keys))
		//fmt.Println("hi")
		return keys[index]

	} else {
		//fmt.Println("hello")
		return keys[0]
	}
}

//Input: method for a node. Takes in the level of the node in the tree, the target
//variable as a string and the dataset as a DataFrame
//Output: no output just sets the attrbutes of the node
func (root *Node) InitializeNode(level int, target string, data DataFrame) {
	root.level = level
	//root.giniIndex = GiniIndex(target, data)
	root.numSamples = data.Nrow()
	root.samplesInCategory = CategoryCounts(target, data)
	//fmt.Println(root.samplesInCategory)
	root.category = ReturnCategory(root.samplesInCategory)
}

// //Input: a method for a randomforest type which is a list of decision trees
// //Output: a slice of strings specifying the predicted values of each row from the input data
// func (forest randForest) Predict(rows DataFrame) []string {
// 	rowPreds := make([]string, rows.Nrow())
//
// 	//iterate thru the rows of the data set
// 	for j := 0; j < rows.Nrow(); j++ {
//
// 		//Select the j-th row
// 		row := rows.Subset(j)
// 		//Initialize a map to store the predictions from each tree
// 		predictions := make(map[string]int)
//
// 		//iterate thru each tree
// 		for i := 0; i < len(forest); i++ {
//
// 			tree := forest[i]
// 			root := tree.Root
//
// 			//while the root is an internal node and not a leaf, leaf does not have a value
// 			//for the condition attribute.
// 			for root.trueChild != nil && root.falseChild != nil {
//
// 				//point to true child
// 				if row.Col(root.conditon.col).Elem(0).Float() >= root.conditon.val {
// 					root = root.trueChild
// 					//point to false child
// 				} else {
// 					root = root.falseChild
// 				}
// 			}
// 			//If the category already a key
// 			if _, found := predictions[root.category]; found {
// 				predictions[root.category] += 1
// 			} else {
// 				predictions[root.category] = 1
// 			}
// 			//not 100% if I should be counting the number of each category from each tree
// 			//then deciding the majority vote or just do it this way
// 		}
// 		rowPreds[j] = ReturnCategory(predictions)
// 	}
// 	return rowPreds
// }

//Input: takes in a min and max int
//Output: builds an array of ints that is incremented by 1 from min to max
func GenerateSeries(min, max int) []int {
	series := make([]int, max-min+1)
	for i := range series {
		series[i] = min + i
	}
	return series
}

//Input: takes in a slice of strings corresponding to the prediction for a sample and
//the data as a DataFrame
//Output: nothing just prints out the confusion matrix
func ConfusionMatrix(testPreds []string, target string, testData DataFrame) {
	uniqueCats := UniqueColValsStr(testData.Col(target))
	targetCol := testData.Col(target)

	numCats := len(uniqueCats)
	matrix := CreateMatrix(numCats, numCats)

	//Obatin the unique categories from the map's keys and sort them
	mapKeys := make([]string, numCats)
	count := 0
	for key, _ := range uniqueCats {
		mapKeys[count] = key
		count++
	}

	sort.Strings(mapKeys)

	for i := 0; i < testData.Nrow(); i++ {
		actual := targetCol.Elem(i).String()
		predicted := testPreds[i]

		aNumCat := FindIndex(actual, mapKeys)
		pNumCat := FindIndex(predicted, mapKeys)

		//fmt.Println(aNumCat, pNumCat)
		matrix[pNumCat][aNumCat] += 1
	}
	//Header
	fmt.Println("")
	fmt.Println(" CONFUSION MATRIX :) ")
	fmt.Println("                                        Actual")
	fmt.Print("              ")

	//Actual Labels
	for i, key := range mapKeys {
		fmt.Print(key + "   ")

		if i == len(mapKeys)-1 {
			fmt.Print("\n")
		}
	}
	//Predicted Labels
	for i, row := range matrix {
		if i == 2 {
			fmt.Print("  Predicted   ")
		}
		for j := range row {
			if i != 2 {
				fmt.Print("              ")
			}
			fmt.Print(i)
			if j == 0 {
				fmt.Print("  ")
			}
			fmt.Print(matrix[i][j])
		}
	}
}

//Input: takes in a feature column as a Series object
//Output: returns a map with strings as keys corresponding to the unique strings(categories)
// in the target column
func UniqueColValsStr(column series.Series) map[string]int {
	uniqueVals := make(map[string]int, column.Len())

	for i := 0; i < column.Len(); i++ {

		category := column.Elem(i).String()
		if _, found := uniqueVals[category]; found != true {
			uniqueVals[category] = 0
		}
	}
	return uniqueVals
}

//Input: desired number of rows and columns to build a matrix
//Output: a matrix that is initialized with zeros
func CreateMatrix(rows, cols int) [][]int {
	matrix := make([][]int, cols)

	for i := 0; i < rows; i++ {
		matrix[i] = make([]int, rows)
	}

	return matrix
}

//Input: a key to locate within an array
//Output: the integer location of the key in the array
func FindIndex(key string, array []string) int {

	for i, val := range array {
		if key == val {
			return i
		}
	}
	return 0
}

//Input: a method for a randomforest type which is a list of decision trees
//Output: a slice of strings specifying the predicted values of each row from the input data
func (forest randForest) PredictRow(row DataFrame) string {

	//Initialize a map to store the predictions from each tree
	predictions := make(map[string]int)

	//iterate thru each tree
	for i := 0; i < len(forest); i++ {

		tree := forest[i]
		root := tree.Root

		//while the root is an internal node and not a leaf, leaf does not have a value
		//for the condition attribute.
		for root.trueChild != nil && root.falseChild != nil {

			//point to true child
			if row.Col(root.conditon.col).Elem(0).Float() >= root.conditon.val {
				root = root.trueChild
				//point to false child
			} else {
				root = root.falseChild
			}
		}
		//If the category already a key
		if _, found := predictions[root.category]; found {
			predictions[root.category] += 1
		} else {
			predictions[root.category] = 1
		}
		//not 100% if I should be counting the number of each category from each tree
		//then deciding the majority vote or just do it this way
	}
	return ReturnCategory(predictions)

}
