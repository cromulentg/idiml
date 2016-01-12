package com.idibon.ml.predict

import scala.collection.mutable

/**
  * Class that stores predict options.
  *
  * It's a case class to make it easy to compare against.
  *
  * @param options the map of options. Casting the value is up to the caller.
  */
case class PredictOptions(options: Map[PredictOption.Value, Any])
//TODO: is there a nicer pattern to get values from this and cast to the right object?

/**
  * Enumeration of options possible during prediction.
  */
object PredictOption extends Enumeration {
  type PredictOption = Value
  val SignificantFeatures, // Shows significant features
    SignificantThreshold // Value a feature needs to be >= if the model understands this value.
    = Value
}

/**
  * Builder class to create a PredictOptions object.
  */
class PredictOptionsBuilder() {
  private val options = new mutable.HashMap[PredictOption.Value, Any]()

  /**
    * If invoked, will ensure that significant features are returned, and
    * if the prediction can make use of it, will limit it to values above
    * the significant threshold passed.
    * @param threshold
    * @return
    */
  def showSignificantFeatures(threshold: Double): PredictOptionsBuilder = {
    options += (PredictOption.SignificantFeatures -> true)
    options += (PredictOption.SignificantThreshold -> threshold)
    this
  }

  def build(): PredictOptions = {
    new PredictOptions(options.toMap)
  }

}
