package com.idibon.ml.predict.ensemble

import com.idibon.ml.alloy.Alloy.{Reader, Writer}
import com.idibon.ml.common.{ArchiveLoader, Engine, Archivable}
import com.idibon.ml.predict._
import com.idibon.ml.predict.ml.TrainingSummary
import org.json4s._

/**
  * Abstract class that houses an ensemble of models.
  *
  * The models map keys aren't used during prediction, but
  * used when saving and loading the models.
  *
  * The number of models doesn't matter. It is up to the subclass
  * to implement reducing results to satisfactory output.
  *
  * E.g. one prediction per label for a document,
  *   or one span prediction/label prediction per span of text
  *
  * @author "Stefan Krawczyk <stefan@idibon.com>" on 3/24/16.
  * @param models
  * @tparam T
  */
abstract class EnsembleModel[T <: PredictResult](models: Map[String, PredictModel[T]])
  extends PredictModel[T]{
  override def getEvaluationMetric(): Double = ???

  /**
    * Returns a training summary. You have to override this to actually return something.
    *
    * @return
    */
  override def getTrainingSummary(): Option[Seq[TrainingSummary]] = {
    val ts = trainingSummary.getOrElse(Seq())
    val summaries = models
      .map({ case (label, model) => model.getTrainingSummary()})
      .collect({ case Some(summary) => summary}).flatten ++ ts
    if (summaries.isEmpty) None else Some(summaries.toSeq)
  }

  /**
    * The method used to predict from a vector of features.
    *
    * @param document Document that contains the original JSON.
    * @param options  Object of predict options.
    * @return
    */
  override def predict(document: Document, options: PredictOptions): Seq[T] = {
    /* augment the document if need be */
    val doc = prePredict(document)
    /* predict against all of the models, concatenate all of the results */
    val predictions = models.par
      .map({ case (label, model) => model.predict(doc, options) })
      .toList.flatten
    /* delegate reducing results to subclasses */
    this.reduce(predictions)
  }

  /**
    * Placeholder function for subclasses to override if they need to.
    *
    * @param document
    * @return
    */
  protected def prePredict(document: Document): Document = document

  /**
    * Function to reduce the output as appropriate to an atomic level
    * for that particular predict type.
    *
    * @param predictions
    * @return
    */
  protected def reduce(predictions: Seq[T]): Seq[T]

  /**
    * Saves the models in this ensemble model to the passed in writer.
    *
    * @param writer destination within Alloy for any resources that
    *   must be preserved for this object to be reloadable.
    * @return JObject containing metadata about the models saved.
    */
  protected def saveModels(writer:Writer): JObject = {
    implicit val formats = org.json4s.DefaultFormats
    // create list of model types by label
    val modelTypes = models.map({case (label, model) => {
      (label, model.reifiedType.getName, model)
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
    JObject(modelMetadata)
  }
}

object EnsembleModel {
  /**
    * Helper method to load models from the passed in reader & model metadata.
    *
    * @param modelNames sequence of string model names.
    * @param modelMetaData archived configuration data returned previous call to saveModels.
    * @param engine implementation of the Engine trait
    * @param reader location within Alloy for loading any resources
    *               previous preserved by a call to
    *               { @link com.idibon.ml.common.Archivable#save}
    * @return
    */
  def load[T <: PredictResult](modelNames: Seq[String],
           modelMetaData: JObject,
           engine: Engine,
           reader: Option[Reader]) = {
    implicit val formats = org.json4s.DefaultFormats
    modelNames.map(name => {
      // get model type
      val modelType =
        Class.forName((modelMetaData \ name \ "class").extract[String])
      // get model metadata JObject
      val indivMeta = (modelMetaData \ name \ "config").extract[Option[JObject]]
      (name, ArchiveLoader
        .reify[PredictModel[T]](modelType, engine, Some(reader.get.within(name)), indivMeta)
        .getOrElse(modelType.newInstance.asInstanceOf[PredictModel[T]]))
    }).toMap
  }
}
