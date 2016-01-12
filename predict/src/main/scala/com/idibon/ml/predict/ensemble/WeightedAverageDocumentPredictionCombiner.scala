package com.idibon.ml.predict.ensemble

import com.idibon.ml.predict.{PredictResultFlag, PredictResult, SingleLabelDocumentResultBuilder, SingleLabelDocumentResult}

/**
  * Class that houses logic to combine document prediction results.
  *
  * Is this named appropriately?
  *
  * Otherwise this class is tightly coupled to the model types. Since
  * we have to special case black & white list rules.
  */
class WeightedAverageDocumentPredictionCombiner(modelIdentifier: String, label: String) {

  /**
    * Method to combine results using the weighted average method.
    *
    * Combines results only on these results:
    *   - SingleLabelDocumentResult objects
    *   - & results that match our label
    *
    * Assumptions:
    * -- Blacklist & Whitelist:
    * - We assume we have an order of preference, and that is the order of the models in the list.
    * --> So if we have two models with a blackOrWhite list, just take the first one.
    * - We don't do anything to the probability, just take it as is.
    * -- General:
    * - For now assume that significant features just get passed through without having to say what
    * model they are from.
    * @param results
    * @return
    */
  def combine(results: List[PredictResult]): SingleLabelDocumentResult = {

    // we only can deal with SingleLabelDocumentResults at the moment & one for the current label
    val singleLabelResults: List[SingleLabelDocumentResult] = results
      .filter(_.isInstanceOf[SingleLabelDocumentResult])
      .map(_.asInstanceOf[SingleLabelDocumentResult])
      .filter(_.label == this.label)

    // deal with special case black/whitelist
    val blackOrWhite = singleLabelResults.filter(_.flags(PredictResultFlag.FORCED))
    // if special case then
    if (blackOrWhite.size > 0) {
      // build new result from old, changing model identifier.
      return new SingleLabelDocumentResultBuilder(modelIdentifier, label)
        .copyFromExistingSingleLabelDocumentResult(blackOrWhite.head)
        .build()
    }
    // sum matchCount
    val matchSum = singleLabelResults.map(_.matchCount).sum
    // compute new weighted 'prob' if matchSum is not 0.0
    val newProb: Float = if (matchSum == 0) 0.0f
    else {
      // sum matchCount * prob
      val weightedSum = singleLabelResults.map(x => x.matchCount * x.probability).sum
      // new prob
      weightedSum / matchSum
    }
    // gather significant features
    val sigFeatures = singleLabelResults.flatMap(_.significantFeatures)
    // add it all & build
    new SingleLabelDocumentResultBuilder(modelIdentifier, label)
      .setProbability(newProb)
      .addSignificantFeatures(sigFeatures)
      .setMatchCount(matchSum)
      .build()
  }
}
