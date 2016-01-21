package com.idibon.ml.predict

/**
  * Class that stores predict options.
  *
  * It's a case class to make it easy to compare against.
  *
  * @param significantFeatureThreshold Default is NaN (off). Otherwise significant features should
  *                                    be returned when the value is not NaN. If the model can use
  *                                    a threshold with significant features, it should use the
  *                                    value here.
  */
case class PredictOptions(significantFeatureThreshold: Float = PredictOptions.NO_FEATURES)

/**
  * Builder class to create a PredictOptions object.
  */
class PredictOptionsBuilder() {
  private var significantFeatureThreshold = PredictOptions.NO_FEATURES
  /**
    * If invoked, will ensure that significant features are returned, and
    * if the prediction can make use of it, will limit it to values above
    * the significant threshold passed.
    * @param threshold
    * @return
    */
  def showSignificantFeatures(threshold: Float): this.type = {
    significantFeatureThreshold = threshold
    this
  }

  def build(): PredictOptions = {
    new PredictOptions(significantFeatureThreshold)
  }

}

object PredictOptions {
  val NO_FEATURES = Float.NaN

  /** The default set of prediction options */
  val DEFAULT = (new PredictOptionsBuilder).build
}
