package com.idibon.ml.feature

import scala.collection.mutable.MutableList

import org.apache.spark.mllib.linalg.Vector

sealed abstract class FeatureGraphBuilder[Repr](name: String) {

  /** Builds the pipeline
    *
    * @param outputNames the names of transform outputs that should be
    *   collected as output for the overall pipeline
    * @return the built FeaturePipeline
    */
  def :=(outputNames: String*): Repr = {
    val transforms = _entries.map(e => (e.name -> e.transformer)).toMap
    val pipeline = (_entries.map(e => new PipelineEntry(e.name, e.inputs))
      :+ (new PipelineEntry(FeatureGraph.OutputStage, outputNames.toList)))

    build(transforms, pipeline)
  }

  def +=(transformer: UnboundPipelineEntry): this.type = {
    _entries += transformer
    this
  }

  def +=(name: String, x: FeatureTransformer, inputs: Seq[String]): this.type = {
    this += UnboundPipelineEntry(name, x, inputs.toList)
  }

  /** Factory for the returned graph object.
    */
  protected def build(transforms: Map[String, FeatureTransformer],
    pipeline: Seq[PipelineEntry]): Repr

  private[this] val _entries = MutableList[UnboundPipelineEntry]()
}

/** Represents an unbound transformer within the builder */
case class UnboundPipelineEntry(name: String,
  transformer: FeatureTransformer,
  inputs: List[String])

/** Builds FeaturePipelines */
class FeaturePipelineBuilder(name: String)
    extends FeatureGraphBuilder[FeaturePipeline](name) {

  protected def build(transforms: Map[String, FeatureTransformer],
    pipeline: Seq[PipelineEntry]):
      FeaturePipeline = {

    FeaturePipeline.bind(transforms, pipeline)
  }
}

object FeaturePipelineBuilder {
  def named(name: String) = new FeaturePipelineBuilder(name)

  def entry(name: String, transformer: FeatureTransformer,
    inputs: String*) = new UnboundPipelineEntry(name, transformer, inputs.toList)
}

/** Builds ChainPipelines */
case class ChainPipelineBuilder(name: String)
    extends FeatureGraphBuilder[ChainPipeline](name) {

  protected def build(transforms: Map[String, FeatureTransformer],
    pipeline: Seq[PipelineEntry]):
      ChainPipeline = {

    val graph = FeatureGraph[Chain[Vector]](name, transforms, pipeline,
      Seq(FeatureGraph.DocumentInput, FeatureGraph.SequenceInput))
    new ChainPipeline(graph, transforms, pipeline)
  }
}


/** Builds SequenceGenerators */
case class SequenceGeneratorBuilder(name: String)
    extends FeatureGraphBuilder[SequenceGenerator](name) {

  protected def build(transforms: Map[String, FeatureTransformer],
    pipeline: Seq[PipelineEntry]):
      SequenceGenerator = {

    val graph = FeatureGraph[Chain[tokenizer.Token]](name, transforms,
      pipeline, Seq(FeatureGraph.DocumentInput))
    new SequenceGenerator(graph, transforms, pipeline)
  }
}
