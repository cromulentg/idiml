package com.idibon.ml.predict.rules

import com.idibon.ml.alloy.Alloy.{Reader, Writer}
import com.idibon.ml.predict.DocumentPredictionResult
import org.apache.spark.mllib.linalg.Vector
import org.json4s.JObject

/**
  * Class taking care of Span rule models. This could become a trait potentially...
  * @param label the index of the label these rules are for
  * @param rules a list of tuples of rule & weights.
  */
class SpanRules(label: Int, rules: List[(String, Double)]) extends RulesModel(label, rules) {
  /**
    * The method used to predict.
    * @param features Vector of features to use for prediction.
    * @return Vector where the index corresponds to a label and the value the
    *         probability for that label.
    */
  override def predict(features: Vector, significantFeatures: Boolean): DocumentPredictionResult = ???

  /**
    * Returns the type of model.
    * @return canonical class name.
    */
  override def getType(): String = this.getClass().getCanonicalName()

  /**
    * The model will use a subset of features passed in. This method
    * should return the ones used.
    * @return Vector (likely SparseVector) where indices correspond to features
    *         that were used.
    */
  // should return the "raw string" feature index.
  override def getFeaturesUsed(): Vector = ???

  /** Reloads the object from the Alloy
    *
    * @param reader location within Alloy for loading any resources
    *               previous preserved by a call to
    *               { @link com.idibon.ml.feature.Archivable#save}
    * @param config archived configuration data returned by a previous
    *               call to { @link com.idibon.ml.feature.Archivable#save}
    * @return this object
    */
  override def load(reader: Reader, config: Option[JObject]): SpanRules.this.type = ???

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
    * THe method used to predict FROM A DOCUMENT!
    * @param document
    * @param significantFeatures
    * @return
    */
  override def predict(document: JObject, significantFeatures: Boolean): DocumentPredictionResult = ???
}
