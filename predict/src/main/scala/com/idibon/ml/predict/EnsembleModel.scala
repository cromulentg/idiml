package com.idibon.ml.predict

import java.io.DataInputStream

import com.idibon.ml.alloy.Alloy.{Writer, Reader}
import org.apache.spark.mllib.linalg.Vector
import org.codehaus.jettison.json.JSONObject
import org.json4s.JObject


/**
  * Ensemble Model is comprised of several models
  * underneath.
  *
  */
class EnsembleModel extends PredictModel {

  var models: List[PredictModel] = null
  var combinerLogic: Any = null //TODO: flesh this out more.

  /**
    * THe method used to predict FROM A DOCUMENT!
    * @param document
    * @param significantFeatures
    * @return
    */
  override def predict(document: JObject, significantFeatures: Boolean): DocumentPredictionResult = ???

  /**
    * The method used to predict.
    * @param features Vector of features to use for prediction.
    * @return Vector where the index corresponds to a label and the value the
    *         probability for that label.
    */
  override def predict(features: Vector, significantFeatures: Boolean): DocumentPredictionResult = {
    //TODO: how to combine results?
    null
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
    *         that were used. Could be a bitvector moving forward, or something
    *         like that.
    */
  // should return the "raw string" feature index.
  override def getFeaturesUsed(): Vector = ???

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

  /** Reloads the object from the Alloy
    *
    * @param reader location within Alloy for loading any resources
    *               previous preserved by a call to
    *               { @link com.idibon.ml.feature.Archivable#save}
    * @param config archived configuration data returned by a previous
    *               call to { @link com.idibon.ml.feature.Archivable#save}
    * @return this object
    */
  override def load(reader: Reader, config: Option[JObject]): EnsembleModel.this.type = ???
}
