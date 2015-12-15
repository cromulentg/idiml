package com.idibon.ml.predict

/**
  * This class houses a span prediction result.
  * It is NOT COMPLETE, and just a placeholder :)
  *
  * @author "Stefan Krawczyk <stefan@idibon.com>"
  *
  */
class SpanPredictionResult{
  //TODO: some fields
}

class SpanPredictionResultBuilder() {

  //TODO: some fields.

  /**
    * Add a prediction result for a span.
    * @param labelIndex
    * @param probability
    * @param tokenOffsetStart the token we want to start at.
    * @param tokenOffsetLength the number of tokens we want to end at.
    * @param tokenCharOffsetStart the charachter we want to start within the starting token.
    * @param tokenCharLength the number of characters we want to end at.
    * @param labelSignificantFeatures
    */
  def addSpanPredictResult(labelIndex: Int, probability: Double, tokenOffsetStart: Int,
                           tokenOffsetLength: Int, tokenCharOffsetStart: Int, tokenCharLength: Int,
                           labelSignificantFeatures: List[(Int, Double)]): Unit = {
    //TODO: figure this part out.
  }

  /**
    * Returns the immutable SpanPredictionResult.
    * @return the prediction result object.
    */
  def build(): SpanPredictionResult = {
    new SpanPredictionResult()
  }
}
