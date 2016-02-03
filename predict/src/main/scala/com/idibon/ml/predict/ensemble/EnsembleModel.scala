package com.idibon.ml.predict.ensemble

import com.idibon.ml.alloy.Alloy.{Reader, Writer}
import com.idibon.ml.common.{Archivable, ArchiveLoader, Engine}
import com.idibon.ml.predict._
import org.apache.spark.mllib.linalg.Vector
import org.json4s._

/**
  * Ensemble Model is comprised of several models underneath.
  *
  * @param label
  * @param models
  */
case class EnsembleModel(label: String, models: List[PredictModel])
    extends PredictModel with Archivable[EnsembleModel, EnsembleModelLoader] {

  val indexToModel = models.zipWithIndex.map({case (model, index) => (index, model)}).toMap

  /**
    * The method used to predict from a FULL DOCUMENT!
    *
    * The lower level models needs to handle "featurization".
    *
    * @param document the JObject to pull from.
    * @param options Object of predict options.
    * @return
    */
  override def predict(document: Document,
                       options: PredictOptions): PredictResult = {
    // delegate to underlying models
    val results: List[(Int, PredictResult)] = indexToModel.par.map(m =>
      (m._1, m._2.predict(document, options))).toList
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
    val combiner = new WeightedAverageDocumentPredictionCombiner("", this.label)
    combiner.combine(results)
  }

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
    val modelClasses = models.map(_.getClass().getName())
    // zip it together
    val modelTriple = (models.indices, models, modelClasses).zipped.toList
    // save each model into it's own space: just use the index as the identifier
    val modelMetadata: List[JField] = modelTriple.map {
      case (index, mod, typ) => {
        val name = index.toString
        JField(name, JObject(List(
          JField("config",
            Archivable.save(mod, writer.within(name)).getOrElse(JNothing)),
          JField("class", JString(typ)))))
      }
    }
    // create the JSON config to return
    val ensembleMetadata = JObject(List(
      JField("label", JString(this.label)),
      JField("size", JInt(this.models.size)),
      JField("model-meta", JObject(modelMetadata))
    ))
    Some(ensembleMetadata)
  }
}

/** Paired loader class for EnsembleModel objects */
class EnsembleModelLoader extends ArchiveLoader[EnsembleModel] {
  /** Reloads the object from the Alloy
    *
    * @param reader location within Alloy for loading any resources
    *               previous preserved by a call to
    *               { @link com.idibon.ml.common.Archivable#save}
    * @param config archived configuration data returned by a previous
    *               call to { @link com.idibon.ml.common.Archivable#save}
    * @return this object
    */
  def load(engine: Engine, reader: Reader, config: Option[JObject]): EnsembleModel = {
    implicit val formats = org.json4s.DefaultFormats
    val label = (config.get \ "label").extract[String]
    val size = (config.get \ "size").extract[Int]
    val modelMeta = (config.get \ "model-meta").extract[JObject]
    val models = (0 until size).map(_.toString).map(name => {
      // get model type
      val modelType =
        Class.forName((modelMeta \ name \ "class").extract[String])
      // get model metadata JObject
      val indivMeta = (modelMeta \ name \ "config").extract[Option[JObject]]
      ArchiveLoader
        .reify[PredictModel](modelType, engine, reader.within(name), indivMeta)
        .getOrElse(modelType.newInstance.asInstanceOf[PredictModel])
    })
    new EnsembleModel(label, models.toList)
  }
}
