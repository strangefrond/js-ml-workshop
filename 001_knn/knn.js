
function KNN(kSize){
	this.kSize = kSize;
	this.points = [];
}

KNN.prototype._distance = function(arr1, arr2){
	return arr1.reduce(function(accum, point, index){
		let diff = (point - arr2[index]);
		return accum + (diff * diff);
	}, 0);
}

KNN.prototype._distances = function(vectorToPredict, trainingData){
	return trainingData.map((dataPt, index) => {
		let [trainVector, type] = dataPt;
		return [this._distance(vectorToPredict, trainVector), type]
	}, KNN);
}

KNN.prototype.train = function(trainingDataArray){
	this.points = [...this.points, ...trainingDataArray];
}

KNN.prototype.predict = function(dataArray){
	return dataArray.map(vector=>this.predictSingle(vector));
}

KNN.prototype._sorted = function(distanceAndTypeArray){
	return distanceAndTypeArray.sort((a, b) => a[0] - b[0]).map(a => a[1]);
}

KNN.prototype._majority = function(k, sortedTypes){
	let nearestK = sortedTypes.slice(0, k);
	let memo = {}
	for (let el of nearestK){
		if (!memo[el]){memo[el] = 1;}
		else {
			memo[el]++;
		}
	}
	let maxOccurrences = 0;
	let maxKey;
	for (let key in memo){
		if (memo[key] > maxOccurrences){
			maxOccurrences = memo[key];
			maxKey = key;
		}
	}
	return +maxKey;
}

KNN.prototype.predictSingle = function(singleDataPoint){
	// get _distances
	// sort them
	// return whatever the majority is of first K sorted
	return this._majority(this.kSize,this._sorted(this._distances(singleDataPoint,this.points)));

}

//takes an array full of 2 value arrays holding a vector and its known type. Checks if the knn function is classifying those data points correctly and gives the knn a score from 0 to 1
KNN.prototype.score = function(knownDataSet){
	let data = knownDataSet.map(a=>a[0]);
	let predictions = this.predict(data);
	let successes = predictions.reduce((accum, prediction, index)=>{
		if (prediction===knownDataSet[index][1]) return accum + 1;
		else return accum;
	}, 0);
	return successes / knownDataSet.length;
}

module.exports = KNN
