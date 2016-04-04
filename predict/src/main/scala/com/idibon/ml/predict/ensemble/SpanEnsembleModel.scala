package com.idibon.ml.predict.ensemble

import com.idibon.ml.alloy.Alloy.{Writer, Reader}
import com.idibon.ml.common.{Engine, ArchiveLoader, Archivable}
import com.idibon.ml.predict._
import org.apache.spark.mllib.linalg.Vector
import org.json4s._

import scala.collection.mutable

/**
  * Span ensemble model implementation.
  *
  * This handles delegating predictions to models as well as reducing results
  * appropriately from the multiple models.
  *
  * @author "Stefan Krawczyk <stefan@idibon.com>" on 3/25/16.
  * @param name
  * @param models
  */
class SpanEnsembleModel(name: String, models: Map[String, PredictModel[Span]])
  extends EnsembleModel[Span](models)
  with Archivable[SpanEnsembleModel, SpanEnsembleModelLoader]{

  override val reifiedType: Class[_ <: PredictModel[Span]] = classOf[SpanEnsembleModel]

  /**
    * Reduces predictions.
    *
    * Each span of text can only have one span associated with it. So reduce
    * any overlaps using a greedy approach.
    *
    * @param predictions the span predictions to reduce
    * @return a sequence of spans that dont overlap
    */
  override protected def reduce(predictions: Seq[Span]): Seq[Span] = {
    // sort by offset, and tie break on negative length
    val sorted = predictions.sortBy(p => (p.offset, -p.length)).toList
    var workspace = sorted
    val mutableList = mutable.ListBuffer[Span]()
    while (workspace.nonEmpty) {
      // grab the head element
      val head = workspace.head
      // find overlapping pieces
      val overlapping = Span.getContiguousOverlappingSpans(head, workspace.tail)
      // send to reducer
      val reduced = Span.greedyReduce(head +: overlapping, Span.chooseSpan)
      // filter < 0.5 Rule Spans & add to mutableList
      mutableList ++= reduced.filterNot(s => s.isRule && s.probability < 0.5f)
      // advance over the list
      workspace = workspace.slice(overlapping.size + 1, workspace.length)
    }
    mutableList.toSeq
  }

  override def getFeaturesUsed(): Vector = ???

  /**
    * Saves the ensemble span model using the passed in writer.
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
      JField("model-meta", modelMetadata)
    ))
    Some(ensembleMetadata)
  }
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
    val name = (config.get \ "name").extract[String]
    val modelNames = (config.get \ "labels").extract[List[String]]
    val modelMeta = (config.get \ "model-meta").extract[JObject]
    val models = EnsembleModel.load(modelNames, modelMeta, engine, reader)
    new SpanEnsembleModel(name, models)
  }
}
