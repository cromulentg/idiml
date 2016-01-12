import scala.collection.mutable.MutableList

package com.idibon.ml.feature {

  /** Builder for FeaturePipelines
    *
    * FeaturePipelines are normally loaded from Alloys; however, for
    * cases where the pipeline must be created programmatically, the
    * builder provides a simpler interface than mocking Alloys and JSON
    * structures.
    */
  class FeaturePipelineBuilder(name: String) {

    /** Builds the pipeline
      *
      * @param outputNames the names of transform outputs that should be
      *   collected as output for the overall pipeline
      * @return the built FeaturePipeline
      */
    def :=(outputNames: String*): FeaturePipeline = {
      val transforms = _entries.map(e => (e.name -> e.transformer)).toMap
      val pipeline = (_entries.map(e => new PipelineEntry(e.name, e.inputs))
        :+ (new PipelineEntry(FeaturePipeline.OutputStage, outputNames.toList)))

      FeaturePipeline.bind(transforms, pipeline)
    }

    /** Adds a transformer to the pipeline
      *
      * @param transformer an Entry containing the name for the transform, the
      *   instance, and all named inputs needed by it
      * @return this
      */
    def +=(transformer: UnboundPipelineEntry): this.type = {
      _entries += transformer
      this
    }

    private[this] val _entries = MutableList[UnboundPipelineEntry]()
  }


  object FeaturePipelineBuilder {
    /** Creates a new builder for a feature pipeline named name
      *
      * @param name the name of the feature pipeline that will be created
      * @return a FeaturePipelineBuilder instance
      */
    def named(name: String) = (new FeaturePipelineBuilder(name))

    def entry(name: String, transformer: FeatureTransformer,
        inputs: String*) = {
      new UnboundPipelineEntry(name, transformer, inputs.toList)
    }
  }

  /** Used to represent an unbound transformer in the builder */
  case class UnboundPipelineEntry(name: String,
    transformer: FeatureTransformer, inputs: List[String])
}
