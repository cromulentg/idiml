package com.idibon.ml.predict.ensemble

import com.idibon.ml.alloy.Alloy.{Reader, Writer}
import com.idibon.ml.common.{Engine, ArchiveLoader, Archivable}
import com.idibon.ml.feature.FeaturePipeline
import com.idibon.ml.predict._
import org.apache.spark.mllib.linalg.Vector
import org.json4s._

import scala.collection.mutable

/**
  * Classification ensemble model implementation.
  *
  * This handles delegating predictions to models as well as reducing results
  * appropriately from the multiple models.
  *
  * @author "Stefan Krawczyk <stefan@idibon.com>"
  * @param name
  * @param models
  * @param featurePipeline
  */
class ClassificationEnsembleModel(name: String,
                                  models: Map[String, PredictModel[Classification]],
                                  override val featurePipeline: Option[FeaturePipeline] = None)
  extends EnsembleModel[Classification](models) with CanHazPipeline
  with Archivable[ClassificationEnsembleModel, ClassificationEnsembleModelLoader]{

  override val reifiedType: Class[_ <: PredictModel[Classification]] = classOf[ClassificationEnsembleModel]

  /**
    * Reduces predictions.
    *
    * Each label can only have one classification associated with it. So reduce by label.
    *
    * @param predictions the classification predictions to reduce
    * @return a sequence of classifications, one for each label
    */
  override protected def reduce(predictions: Seq[Classification]): Seq[Classification] = {
    predictions
      .groupBy(_.label)
      .map({ case (label, components) => Classification.reduce(components) }).toSeq
      .sortWith(_.probability > _.probability)
  }

  override def getFeaturesUsed(): Vector = ???

  /**
    * Saves the ensemble classification model using the passed in writer.
    *
    * @param writer destination within Alloy for any resources that
    *   must be preserved for this object to be reloadable
    * @return Some[JObject] of configuration data that must be preserved
    *   to reload the object. None if no configuration is needed
    */
  def save(writer: Writer): Option[JObject] = {
    implicit val formats = org.json4s.DefaultFormats
    //save each model into it's own space & get model metadata
    val modelMetadata = this.saveModels(writer)
    val modelNames = JArray(models.map({case (label, _) => JString(label)}).toList)
    // create JSON config to return
    val ensembleMetadata = JObject(List(
      JField("name", JString(name)),
      JField("labels", modelNames),
      JField("model-meta", modelMetadata),
      savePipelineIfPresent(writer)
    ))
    Some(ensembleMetadata)
  }
}


class ClassificationEnsembleModelLoader extends ArchiveLoader[ClassificationEnsembleModel] {
  /** Reloads the object from the Alloy
    *
    * @param engine implementation of the Engine trait
    * @param reader location within Alloy for loading any resources
    *               previous preserved by a call to
    *               { @link com.idibon.ml.common.Archivable#save}
    * @param config archived configuration data returned by a previous
    *               call to { @link com.idibon.ml.common.Archivable#save}
    * @return this object
    */
  override def load(engine: Engine,
                    reader: Option[Reader],
                    config: Option[JObject]): ClassificationEnsembleModel = {
    implicit val formats = org.json4s.DefaultFormats
    val name = (config.get \ "name").extract[String]
    val modelNames = (config.get \ "labels").extract[List[String]]
    val modelMeta = (config.get \ "model-meta").extract[JObject]
    val pipeline = CanHazPipeline.loadPipelineIfPresent(engine, reader, config)
    val models = EnsembleModel.load[Classification](modelNames, modelMeta, engine, reader)
    new ClassificationEnsembleModel(name, models, pipeline)
  }
}
