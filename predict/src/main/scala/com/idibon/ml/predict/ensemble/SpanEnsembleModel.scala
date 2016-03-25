package com.idibon.ml.predict.ensemble

import com.idibon.ml.alloy.Alloy.{Reader}
import com.idibon.ml.common.{Engine, ArchiveLoader, Archivable}
import com.idibon.ml.feature.FeaturePipeline
import com.idibon.ml.predict._
import org.apache.spark.mllib.linalg.Vector
import org.json4s._

/**
  * Span ensemble model implementation.
  *
  * @author "Stefan Krawczyk <stefan@idibon.com>" on 3/25/16.
  *
  * @param models
  * @param featurePipeline
  */
class SpanEnsembleModel(models: Map[String, PredictModel[Span]],
                        override val featurePipeline: Option[FeaturePipeline] = None)
  extends EnsembleModel[Span](models, featurePipeline)
  with Archivable[SpanEnsembleModel, SpanEnsembleModelLoader]{

  override val reifiedType: Class[_ <: PredictModel[Span]] = classOf[SpanEnsembleModel]

  /** Used to reduce multiple Classifications for a label into a final result */
  private [this] val _reducer: PredictResultReduction[Span] = ???

  /**
    * Reduces predictions
    * @param predictions
    * @return
    */
  override protected def reduce(predictions: Seq[Span]): Seq[Span] = ???

  override def getFeaturesUsed(): Vector = ???
}

class SpanEnsembleModelLoader extends ArchiveLoader[SpanEnsembleModel] {
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
                    config: Option[JObject]): SpanEnsembleModel = {
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
        .reify[PredictModel[Span]](modelType, engine, Some(reader.get.within(name)), indivMeta)
        .getOrElse(modelType.newInstance.asInstanceOf[PredictModel[Span]]))
    }).toMap
    new SpanEnsembleModel(models, pipeline)
  }
}
