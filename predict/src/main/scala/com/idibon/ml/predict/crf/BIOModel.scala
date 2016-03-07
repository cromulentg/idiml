package com.idibon.ml.predict.crf

import scala.util.Try

import com.idibon.ml.predict._
import com.idibon.ml.feature._
import com.idibon.ml.common.{Archivable, ArchiveLoader, Engine}
import com.idibon.ml.alloy.Alloy

import org.apache.spark.mllib.linalg.Vector
import com.typesafe.scalalogging.StrictLogging
import org.json4s._

/** Performs entity recognition using ConLL-style B/I/O sequence tags
  *
  * Given feature graphs which
  *   1) convert a document into a chain of tokens, and
  *   2) convert a chain of tokens into a chain of feature lists,
  * and
      3) a FACTORIE model that infers the highest-likelihood sequence of
  *      B_$label, I_$label and OUTSIDE tags for a chain of features
  *
  * extracts the entities for each label from the document
  */
class BIOModel(model: FactorieCRF,
  sequencer: SequenceGenerator,
  extractor: ChainPipeline) extends PredictModel[Span]
    with Archivable[BIOModel, BIOModelLoader]
    with BIOAssembly {

  /** Perform a Span prediction for the document */
  def predict(doc: Document, options: PredictOptions): Seq[Span] = {
    /* TODO: apply language-detection and content-detection once and store
     * within a mutated document object for the pipelines */
    val tokens = sequencer(doc.json)
    val tagsWithConfidence = model.predict(extractor(doc.json, tokens).flatten)
    assemble(tokens.flatten, tagsWithConfidence)
  }

  def getEvaluationMetric(): Double = ???

  def getFeaturesUsed(): Vector = ???

  /** Representative class to use for loading and saving within alloy */
  val reifiedType = classOf[BIOModel]

  /** Saves this model to an alloy
    *
    * @param writer location within alloy to save model
    */
  def save(writer: Alloy.Writer): Option[JObject] = {
    val resource = writer.resource(BIOModel.CRF_RESOURCE)
    try {
      model.serialize(resource)
    } finally {
      resource.close()
    }
    Some(JObject(List(
      JField(BIOModel.SEQUENCE_GENERATOR,
        sequencer.save(writer.within(BIOModel.SEQUENCE_GENERATOR)).getOrElse(JNothing)),
      JField(BIOModel.FEATURE_EXTRACTOR,
        extractor.save(writer.within(BIOModel.FEATURE_EXTRACTOR)).getOrElse(JNothing)),
      JField("version", JInt(BIOModel.VERSION)))))
  }
}

object BIOModel {
  val CRF_RESOURCE = "crf.dat"
  val SEQUENCE_GENERATOR = "sequenceGenerator"
  val FEATURE_EXTRACTOR = "featureExtractor"
  val VERSION_1 = 1

  val VERSION = VERSION_1
}

/** Paired loader class for BIOModels */
class BIOModelLoader extends ArchiveLoader[BIOModel] {

  /** Loads a BIOModel from an Alloy */
  def load(engine: Engine, reader: Option[Alloy.Reader],
    config: Option[JObject]): BIOModel = {

    implicit val formats = org.json4s.DefaultFormats

    val generator = (new SequenceGeneratorLoader).load(engine,
      reader.map(_.within(BIOModel.SEQUENCE_GENERATOR)),
      config.flatMap(c => (c \ BIOModel.SEQUENCE_GENERATOR).extract[Option[JObject]]))

    val extractor = (new ChainPipelineLoader).load(engine,
      reader.map(_.within(BIOModel.FEATURE_EXTRACTOR)),
      config.flatMap(c => (c \ BIOModel.FEATURE_EXTRACTOR).extract[Option[JObject]]))

    val crf = reader.map(r => {
      val resource = r.resource(BIOModel.CRF_RESOURCE)
      val model = Try(FactorieCRF.deserialize(resource))
      resource.close()
      model.get
    }).get

    new BIOModel(crf, generator, extractor)
  }
}
