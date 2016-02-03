package com.idibon.ml.predict.ensemble

import com.idibon.ml.alloy.Alloy.{Reader, Writer}
import com.idibon.ml.common.{Archivable, ArchiveLoader, Engine}
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
case class GangModel(multiLabelModel: PredictModel, models: Map[String, PredictModel])
  extends PredictModel with Archivable[GangModel, GangModelLoader] {
  /**
    * The method used to predict from a vector of features.
    *
    * @param features Vector of features to use for prediction.
    * @param options  Object of predict options.
    * @return
    */
  override def predict(features: Vector, options: PredictOptions): PredictResult = {
    val mldr: MultiLabelDocumentResult = multiLabelModel
      .predict(features, options)
      .asInstanceOf[MultiLabelDocumentResult]
    val labelResults = models.par
      .map({ case (label, model) =>
        (label, model.predict(features, options).asInstanceOf[SingleLabelDocumentResult])})
      .toList.toMap
    combineResults(mldr, labelResults)
  }

  /**
    * Helper method to combine results.
    *
    * @param multiResult
    * @param singleResults
    * @return
    */
  def combineResults(multiResult: MultiLabelDocumentResult, singleResults: Map[String, SingleLabelDocumentResult]): PredictResult = {

    // for each label in the multi result - just do single label result and use that combiner
    val labelResults = multiResult.par.map({case sldr => {
      val label: String = sldr.label
      if (singleResults.get(label).isDefined){
        new WeightedAverageDocumentPredictionCombiner(this.getType(), label)
          .combine(List(sldr, singleResults(label)))
      } else {
        sldr
      }
    }})// create sorted list by probability (but sorted by label for tie breaking)
      .toList.sortWith(_.label < _.label).sortWith(_.probability > _.probability)
      // now create label, single label result tuple.
      .map({case sldr => {
      (sldr.getLabel(),
        new SingleLabelDocumentResultBuilder(
          this.getType(), sldr.getLabel()).copyFromExistingSingleLabelDocumentResult(sldr))
    }}).toMap
    new MultiLabelDocumentResultBuilder(this.getType(), labelResults).build()
  }

  /**
    * Returns the type of model.
    *
    * @return canonical class name.
    */
  override def getType(): String = this.getClass().getName()

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
  override def predict(document: JObject, options: PredictOptions): PredictResult = {
    val mldr: MultiLabelDocumentResult = multiLabelModel
      .predict(document, options)
      .asInstanceOf[MultiLabelDocumentResult]
    val labelResults = models.par
      .map({ case (label, model) =>
        (label, model.predict(document, options).asInstanceOf[SingleLabelDocumentResult])})
      .toList.toMap
    combineResults(mldr, labelResults)
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
    val modelTypes = (List((GangModel.MULTI_CLASS_LABEL, multiLabelModel.getClass.getName, multiLabelModel))
      ++ models.map({case (label, model) => (label, model.getClass.getName, model)})).toList
    //save each model into it's own space
    val modelMetadata: List[JField] = modelTypes.map {
      case (label, typ, model) => {
        JField(label, JObject(List(
          JField("config",
            Archivable.save(model, writer.within(label)).getOrElse(JNothing)),
          JField("class", JString(typ)))))
      }
    }
    val labels = JArray(
      List(JString(GangModel.MULTI_CLASS_LABEL)) ++ models.map({case (label, _) => JString(label)}))
    // create JSON config to return
    val gangMetadata = JObject(List(
      JField("labels", labels),
      JField("model-meta", JObject(modelMetadata))
    ))
    Some(gangMetadata)
  }
}

object GangModel {
  val MULTI_CLASS_LABEL = "\u000all"
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
  override def load(engine: Engine, reader: Reader, config: Option[JObject]): GangModel = {
    implicit val formats = org.json4s.DefaultFormats
    val labels = (config.get \ "labels").extract[List[String]]
    val modelMeta = (config.get \ "model-meta").extract[JObject]
    val models = labels.map(name => {
      // get model type
      val modelType =
        Class.forName((modelMeta \ name \ "class").extract[String])
      // get model metadata JObject
      val indivMeta = (modelMeta \ name \ "config").extract[Option[JObject]]
      (name, ArchiveLoader
        .reify[PredictModel](modelType, engine, reader.within(name), indivMeta)
        .getOrElse(modelType.newInstance.asInstanceOf[PredictModel]))
    }).toMap
    new GangModel(models(GangModel.MULTI_CLASS_LABEL),
      models.filter(x => !x._1.equals(GangModel.MULTI_CLASS_LABEL)))
  }
}
