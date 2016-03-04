package com.idibon.ml.feature

import com.idibon.ml.alloy.Alloy
import com.idibon.ml.common.{Archivable, ArchiveLoader, Engine}

import com.typesafe.scalalogging.StrictLogging

import org.json4s.JObject

/** Converts a document into a chain of tokens suitable for sequence models
  *
  * Sequence classifier models assign tags to a sequence of elements,
  * represented as a Chain[Feature[_]]. Prior to applying sequence
  * classification to a document, the document must be partitioned into
  * a sequence of tokens with a fixed, unique position within the original
  * text, and each token in the Chain must be converted to a list of features.
  */
class SequenceGenerator(
  graph: FeatureGraph,
  transforms: Map[String, FeatureTransformer],
  pipeline: Seq[PipelineEntry])
    extends Archivable[SequenceGenerator, SequenceGeneratorLoader]
    with Function1[JObject, Chain[tokenizer.Token]]
    with Freezable[SequenceGenerator]
    with StrictLogging {

  /** Processes the document using the bound feature graph
    */
  def apply(document: JObject): Chain[tokenizer.Token] = {
    val transformed = graph(scala.collection.mutable.Map[String, Any](
      FeatureGraph.DocumentInput -> document))

    /* FIXME: what would it mean to return multiple outputs here? this
     * doesn't seem to make a ton of sense, but it's possible that there
     * is some interesting use case (variable tokenization schemes?) that
     * might be possible */
    transformed(FeatureGraph.OutputStage)
      .asInstanceOf[Seq[Chain[tokenizer.Token]]]
      .head
  }

  /** Saves the sequence generator and all resources to an alloy
    *
    * @param writer output alloy interface
    * @return configuration JSON for the sequence generator
    */
  def save(writer: Alloy.Writer): Option[JObject] = {
    FeatureGraph.saveGraph(writer, transforms, pipeline)
  }

  /** Freezes the graph and all transforms in it
    *
    * @return frozen SequenceGenerator
    */
  def freeze(): SequenceGenerator = {
    val frozen = transforms.map(_ match {
      case (n, x: Freezable[_]) => (n -> x.freeze.asInstanceOf[FeatureTransformer])
      case other => other
    }).toMap

    val frozenGraph = FeatureGraph[Chain[tokenizer.Token]](
      graph.name, frozen, pipeline, Seq(FeatureGraph.DocumentInput))

    new SequenceGenerator(frozenGraph, frozen, pipeline)
  }
}

/** Paired loader class for SequenceGenerators */
class SequenceGeneratorLoader
    extends ArchiveLoader[SequenceGenerator]
    with FeatureGraphLoader {
  /** Loads the sequence generator
    *
    * @param engine current engine context
    * @param reader alloy reader for loading graph resources
    * @param config pipeline configuration json
    */
  def load(engine: Engine, reader: Option[Alloy.Reader],
    config: Option[JObject]): SequenceGenerator = {

    val transforms = loadTransformers(engine, reader, config)
    val dependencies = loadPipelineEntries(config)

    val graph = FeatureGraph[Chain[tokenizer.Token]](
      FeaturePipeline.DefaultName, transforms, dependencies,
      Seq(FeatureGraph.DocumentInput))

    val sequencer = new SequenceGenerator(graph, transforms, dependencies)
    if (reader.isDefined) sequencer.freeze() else sequencer
  }
}
