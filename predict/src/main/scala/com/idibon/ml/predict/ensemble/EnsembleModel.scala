package com.idibon.ml.predict.ensemble

import com.idibon.ml.alloy.Alloy.{Writer}
import com.idibon.ml.common.{Archivable}
import com.idibon.ml.feature.FeaturePipeline
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
  *
  * @param models
  * @param featurePipeline
  * @tparam T
  */
abstract class EnsembleModel[T <: PredictResult](
  models: Map[String, PredictModel[T]],
  override val featurePipeline: Option[FeaturePipeline] = None)
  extends PredictModel[T] with CanHazPipeline {
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
    /* if a feature pipeline exists for this gang model, apply it to the
     * document and pass the results and the inversion function to the
     * subordinate models */
    val doc = applyPipelineIfPresent(document)
    /* predict against all of the models, concatenate all of the results */
    val predictions = models.par
      .map({ case (label, model) => model.predict(doc, options) })
      .toList.flatten
    /* delegate reducing results to subclasses */
    this.reduce(predictions)
  }

  protected def reduce(predictions: Seq[T]): Seq[T]

  /** Serializes the object within the Alloy
    *
    * This is here because it's the same for the base classes.
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
  def save(writer: Writer): Option[JObject] = {
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
    val labels = JArray(models.map({case (label, _) => JString(label)}).toList)
    // create JSON config to return
    val ensembleMetadata = JObject(List(
      JField("labels", labels),
      JField("model-meta", JObject(modelMetadata)),
      savePipelineIfPresent(writer)
    ))
    Some(ensembleMetadata)
  }
}
