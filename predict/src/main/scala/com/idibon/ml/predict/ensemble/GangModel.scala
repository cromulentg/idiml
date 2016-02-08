package com.idibon.ml.predict.ensemble

import com.idibon.ml.alloy.Alloy.{Reader, Writer}
import com.idibon.ml.common.{Archivable, ArchiveLoader, Engine}
import com.idibon.ml.feature.FeaturePipeline
import com.idibon.ml.predict._
import org.apache.spark.mllib.linalg.Vector
import org.json4s._

/**
  * A Gang Model is our term for describing handling a multi-class model with per label models.
  *
  * Usually this will manifest itself as 1) multinomial logistic regression model with a few
  * per label rule models.
  *
  *
  * @param multiLabelModel This is the main action piece. Needs to return MultiLabelDocumentResult.
  * @param models Map of label -> model. Okay to be empty.
  */
case class GangModel(
  models: Map[String, PredictModel[Classification]],
  override val featurePipeline: Option[FeaturePipeline] = None)
    extends PredictModel[Classification] with CanHazPipeline
    with Archivable[GangModel, GangModelLoader] {

  /** Used to reduce multiple Classifications for a label into a final result */
  private [this] val _reducer: PredictResultReduction[Classification] = Classification

  /**
    * The model will use a subset of features passed in. This method
    * should return the ones used.
    *
    * @return Vector (likely SparseVector) where indices correspond to features
    *         that were used.
    */
  override def getFeaturesUsed(): Vector = ???

  /**
    * The method used to predict from a FULL DOCUMENT!
    *
    * The model needs to handle "featurization" here.
    *
    * @param document the JObject to pull from.
    * @param options  Object of predict options.
    * @return
    */
  def predict(input: Document, options: PredictOptions): Seq[Classification] = {
    /* if a feature pipeline exists for this gang model, apply it to the
     * document and pass the results and the inversion function to the
     * subordinate models */
    val document = applyPipelineIfPresent(input)

    /* classify against all of the models, concatenate all of the results,
     * then group all of the partial results for each label together and
     * pass them to the reducer to compute the final result. return the
     * results for all labels sorted by probability */
    models.par.map({ case (label, model) => model.predict(document, options) })
      .toList.flatten.groupBy(_.label)
      .map({ case (label, components) => _reducer.reduce(components) }).toSeq
      .sortWith(_.probability > _.probability)
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
  override def save(writer: Writer): Option[JObject] = {
    implicit val formats = org.json4s.DefaultFormats
    // create list of model types by label
    val modelTypes = models.map({case (label, model) => {
      (label, model.getClass.getName, model)
    }}).toList
    //save each model into it's own space
    val modelMetadata: List[JField] = modelTypes.map {
      case (label, typ, model) => {
        JField(label, JObject(List(
          JField("config",
            Archivable.save(model, writer.within(label)).getOrElse(JNothing)),
          JField("class", JString(typ)))))
      }
    }
    val labels = JArray(models.map({case (label, _) => JString(label)}).toList)
    // create JSON config to return
    val gangMetadata = JObject(List(
      JField("labels", labels),
      JField("model-meta", JObject(modelMetadata)),
      savePipelineIfPresent(writer)
    ))
    Some(gangMetadata)
  }
}

/** Paired loader class for EnsembleModel objects */
class GangModelLoader extends ArchiveLoader[GangModel] {
  /** Reloads the object from the Alloy
    *
    * @param engine implementation of the Engine trait
    * @param reader location within Alloy for loading any resources
    *               previous preserved by a call to
    *               { @link com.idibon.ml.feature.Archivable#save}
    * @param config archived configuration data returned by a previous
    *               call to { @link com.idibon.ml.feature.Archivable#save}
    * @return this object
    */
  override def load(engine: Engine, reader: Option[Reader],
      config: Option[JObject]): GangModel = {

    implicit val formats = org.json4s.DefaultFormats
    val labels = (config.get \ "labels").extract[List[String]]
    val modelMeta = (config.get \ "model-meta").extract[JObject]

    val pipeline = CanHazPipeline.loadPipelineIfPresent(engine, reader, config)

    val models = labels.map(name => {
      // get model type
      val modelType =
        Class.forName((modelMeta \ name \ "class").extract[String])
      // get model metadata JObject
      val indivMeta = (modelMeta \ name \ "config").extract[Option[JObject]]
      (name, ArchiveLoader
        .reify[PredictModel[Classification]](modelType, engine, Some(reader.get.within(name)), indivMeta)
        .getOrElse(modelType.newInstance.asInstanceOf[PredictModel[Classification]]))
    }).toMap
    new GangModel(models, pipeline)
  }
}
