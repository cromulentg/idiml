package com.idibon.ml.predict.ml


import com.idibon.ml.alloy.Alloy.{Writer, Reader}
import com.idibon.ml.predict.{PredictOptions, SingleLabelDocumentResultBuilder, PredictResult}
import com.idibon.ml.feature.{Archivable, ArchiveLoader}
import org.apache.spark.ml.classification.IdibonSparkLogisticRegressionModelWrapper
import org.apache.spark.mllib.linalg.Vector
import org.json4s.JObject

/**
  * @author "Stefan Krawczyk <stefan@idibon.com>"
  *
  * This class implements our LogisticRegressionModel.
  */
class IdibonLogisticRegressionModel extends MLModel
  with Archivable[IdibonLogisticRegressionModel, IdibonLogisticRegressionModelLoader] {


  var lrm: IdibonSparkLogisticRegressionModelWrapper = null

  /**
    * The method used to predict from a FULL DOCUMENT!
    *
    * The model needs to handle "featurization" here.
    *
    * @param document the JObject to pull from.
    * @param options Object of predict options.
    * @return
    */
  override def predict(document: JObject,
                       options: PredictOptions): PredictResult = ???

  /**
    * The method used to predict from a vector of features.
    * @param features Vector of features to use for prediction.
    * @param options Object of predict options.
    * @return
    */
  override def predict(features: Vector,
                       options: PredictOptions): PredictResult = {
    val results: Vector = lrm.predictProbability(features)
    val builder = new SingleLabelDocumentResultBuilder(this.getType(), "")
    for (labelIndex <- 0 until results.size) {
      // spark uses double underneath...
      builder.setProbability(results.apply(labelIndex).toFloat)
      if (!options.significantFeatureThreshold.isNaN()) {
        // TODO: get significant features.
        // e.g. we need to find all the features that have a weight above X.
        // We should also check whether we need to take any transform on the weights. e.g. exp ?
      }
    }
    builder.build()
  }

  /**
    * Returns the type of model. Perhaps this should be an enum?
    * @return
    */
  override def getType(): String = this.getClass().getCanonicalName()

  /**
    * The model will use a subset of features passed in. This method
    * should return the ones used.
    * @return Vector (likely SparseVector) where indices correspond to features
    *         that were used.
    */
  override def getFeaturesUsed(): Vector = {
    //For the predict case this is fine I believe, since we should not have 0 valued features.
    lrm.weights
  }

  /** Serializes the object within the Alloy
    *
    * Implementations are responsible for persisting any internal state
    * necessary to re-load the object (for example, feature-to-vector
    * index mappings) to the provided Alloy.Writer.
    *
    * Implementations may return a JObject of configuration data
    * to include when re-loading the object.
    *
    * @param writer destination within Alloy for any resources that
    *               must be preserved for this object to be reloadable
    * @return Some[JObject] of configuration data that must be preserved
    *         to reload the object. None if no configuration is needed
    */
  override def save(writer: Writer): Option[JObject] = ???

  /**
    * Override equals so that we can make unit tests simpler.
    * @param that
    * @return
    */
  override def equals(that: scala.Any): Boolean = {
    that match {
      case that: IdibonLogisticRegressionModel => {
        this.lrm.equals(that.lrm)
      }
      case _ => false
    }
  }
}

/** Paired loader class for IdibonLogisticRegressionModel instances */
class IdibonLogisticRegressionModelLoader
    extends ArchiveLoader[IdibonLogisticRegressionModel] {

  /** Reloads the object from the Alloy
    *
    * @param reader location within Alloy for loading any resources
    *               previous preserved by a call to
    *               { @link com.idibon.ml.feature.Archivable#save}
    * @param config archived configuration data returned by a previous
    *               call to { @link com.idibon.ml.feature.Archivable#save}
    * @return this object
    */
  def load(reader: Reader, config: Option[JObject]): IdibonLogisticRegressionModel = ???
}
