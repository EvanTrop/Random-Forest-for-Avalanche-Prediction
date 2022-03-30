//Evan Trop, etrop
//Final Project : Predicting Avalanche Problems with the Random Forest algorithm
//December 11,2020
package main

type randForest []*Tree

type Tree struct {
	minNumLeaves    int
	maxDepth        int
	minSamplesSplit int
	Root            *Node
}

type Node struct {
	//label                 string
	conditon              Condition
	level                 int
	giniIndex             float64
	numSamples            int
	category              string
	samplesInCategory     map[string]int
	falseChild, trueChild *Node
}

type Condition struct {
	col string
	val float64
}
