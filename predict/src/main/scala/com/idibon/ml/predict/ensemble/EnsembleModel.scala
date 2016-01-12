package com.idibon.ml.predict.ensemble

import com.idibon.ml.alloy.Alloy.{Reader, Writer}
import com.idibon.ml.predict.{PredictOptions, PredictResult, SingleLabelDocumentResult, PredictModel}
import org.apache.spark.mllib.linalg.Vector
import org.json4s._

/**
  * Ensemble Model is comprised of several models underneath.
  *
  * @param label
  * @param models
  */
class EnsembleModel(var label: String, var models: List[PredictModel]) extends PredictModel {

  /**
    * Constructor for making load easy.
    */
  def this() {
    this("", List())
  }

  /**
    * The method used to predict from a FULL DOCUMENT!
    *
    * The lower level models needs to handle "featurization".
    *
    * @param document the JObject to pull from.
    * @param options Object of predict options.
    * @return
    */
  override def predict(document: JObject,
                       options: PredictOptions): PredictResult = {
    // create list of (index, model)
    val zipped = (models.indices, models).zipped.toList
    // delegate to underlying models
    val results: List[(Int, PredictResult)] = zipped.par.map(m =>
      (m._1, m._2.predict(document, options))).toList
    // Reorder list to match models list since we ran in parallel, order isn't guaranteed
    combineResults(results.sortBy(_._1).map(_._2))
  }

  /**
    * The method used to predict from a vector of features.
    * @param features Vector of features to use for prediction.
    * @param options Object of predict options.
    * @return
    */
  override def predict(features: Vector,
                       options: PredictOptions): PredictResult = {
    // create list of (index, model)
    val zipped = (models.indices, models).zipped.toList
    // delegate to underlying models
    val results: List[(Int, PredictResult)] = zipped.par.map(m =>
      (m._1, m._2.predict(features, options))).toList
    // Reorder list to match models list since we ran in parallel, order isn't guaranteed
    combineResults(results.sortBy(_._1).map(_._2))
  }

  /**
    * Helper method to use the combiner and combine the results.
    * @param results
    * @return
    */
  def combineResults(results: List[PredictResult]): SingleLabelDocumentResult = {
    // combine results -- hardcoded object here for now
    val combiner = new WeightedAverageDocumentPredictionCombiner(this.getType(), this.label)
    // only want to deal with single label type so, cast all objects

    combiner.combine(results)
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
  // TODO: decide whether this still makes sense.
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
  override def save(writer: Writer): Option[JObject] = {
    implicit val formats = org.json4s.DefaultFormats
    // create list of model types
    val modelClasses = models.map(_.getClass().getCanonicalName())
    // zip it together
    val modelTriple = (models.indices, models, modelClasses).zipped.toList
    // save each model into it's own space: just use the index as the identifier
    val modelMetadata: List[JField] = modelTriple.map {
      case (index, mod, typ) => {
        // create the space
        val modWriter = writer.within(index.toString)
        // save the model & create tuple of index -> metadataconfig
        JField(index.toString(), mod.save(modWriter).getOrElse(JNothing))
      }
    }
    // create map to tell us what model type is at what index
    val indexToModelType: List[JField] = modelTriple.map {
      case (index, mod, typ) => {
        JField(index.toString(), JString(typ))
      }
    }
    // create the JSON config to return
    val ensembleMetadata = JObject(List(
      JField("label", JString(this.label)),
      JField("size", JInt(this.models.size)),
      JField("model-meta", JObject(modelMetadata)),
      JField("model-index", JObject(indexToModelType))
    ))
    Some(ensembleMetadata)
  }

  /** Reloads the object from the Alloy
    *
    * @param reader location within Alloy for loading any resources
    *               previous preserved by a call to
    *               { @link com.idibon.ml.feature.Archivable#save}
    * @param config archived configuration data returned by a previous
    *               call to { @link com.idibon.ml.feature.Archivable#save}
    * @return this object
    */
  override def load(reader: Reader, config: Option[JObject]): EnsembleModel.this.type = {
    implicit val formats = org.json4s.DefaultFormats
    this.label = (config.get \ "label").extract[String]
    val size = (config.get \ "size").extract[Int]
    var models = List[PredictModel]()
    for (i <- 0 until size) {
      // get model metadata JObject
      val modelMeta = (config.get \ "model-meta" \ i.toString).extract[JObject]
      // get model type
      val modelType = (config.get \ "model-index" \ i.toString).extract[String]
      // create reader for them from the appropriate place
      val modelReader = reader.within(i.toString)
      // create model and load
      val modelInstance = Class.forName(modelType)
        .newInstance().asInstanceOf[PredictModel].load(modelReader, Some(modelMeta))
      // append to models list
      models = models ::: List(modelInstance)
    }
    this.models = models
    this
  }

  /**
    * Override equals so that we can make unit tests simpler.
    * @param that
    * @return
    */
  override def equals(that: scala.Any): Boolean = {
    that match {
      case that: EnsembleModel => {
        this.label == that.label && this.models.equals(that.models)
      }
      case _ => false
    }
  }
}
