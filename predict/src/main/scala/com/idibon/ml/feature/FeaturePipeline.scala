package com.idibon.ml.feature

import com.typesafe.scalalogging.{StrictLogging, Logger}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.json4s._

import scala.Boolean
import scala.collection.mutable.{HashSet => MutableSet, ListBuffer}
import scala.collection.{Map => AnyMap}
import scala.collection.mutable

import com.idibon.ml.alloy.Alloy
import com.idibon.ml.common.{ArchiveLoader, Archivable, Engine}


/** The feature pipeline transforms documents into feature vectors
  */
class FeaturePipeline(
  state: LoadState,
  outputDimensions: Option[Seq[(TerminableTransformer, Int)]] = None)
    extends Archivable[FeaturePipeline, FeaturePipelineLoader]
    with Freezable[FeaturePipeline]
    with Function1[JObject, Vector]
    with StrictLogging {

  val totalDimensions = outputDimensions.map(seq => seq.map(_._2).sum)

  /* store the sum of all output dimensions of the ordered output
   * transforms preceding each transform, for quick mapping from
   * the global feature dimensions to each transform's native space */
  val priorDimensions = outputDimensions.map(dims => {
    dims.foldLeft(List(0))((sums, dim) => dim._2 + sums.head :: sums).tail.reverse
  })

  val isFrozen = outputDimensions.isDefined

  /**
    * Function to prime the pipeline.
    *
    * @param documents
    */
  def prime(documents: TraversableOnce[JObject]): FeaturePipeline = {
    if (isFrozen) throw new IllegalStateException("Pipeline is frozen")
    for (document <- documents) {
      applyFeatureGraph(document)
    }
    freeze()
  }

  /**
    * Helper method to effectively freeze the state of the pipeline.
    *
    * It saves the outputStage to a list, as well as the total dimensions and
    * the size of each outputStage's dimensions.
    */
  def freeze(): FeaturePipeline = {
    if (isFrozen) return this

    /* freeze all of the freezable transforms, updating the transforms map
     * with the modified version */
    val frozen = this.state.transforms.map(_ match {
      case (n, x: Freezable[_]) => (n -> x.freeze.asInstanceOf[FeatureTransformer])
      case other => other
    }).toMap

    // re-bind the graph with the frozen transforms
    val frozenGraph = FeaturePipeline.bindGraph(frozen, this.state.pipeline)

    // and compute the dimensionality of each output
    val terminables = FeaturePipeline
      .collectOutputs(frozen, this.state.pipeline)
      .map({ case (_, xf) => xf.asInstanceOf[TerminableTransformer] })

    new FeaturePipeline(
      new LoadState(frozenGraph, this.state.pipeline, frozen),
      Some(terminables.map(t => {
        (t, t.numDimensions
          .getOrElse(throw new IllegalStateException("Invalid dimensions")))
      })))
  }

  /**
    * Helper method to apply the feature graph to a document.
    *
    * @param document
    * @return sequence of $output vectors
    */
  def applyFeatureGraph(document: JObject): mutable.Map[String, Any] = {
    /* the intermediate data passed between pipeline stages; initialize
         * with the document under the reserved name "$document" */
    val intermediates = mutable.Map[String, Any](
      FeatureGraph.DocumentInput -> document
    )
    this.state.graph(intermediates)
  }

  /** Applies the entire feature pipeline to the provided document.
    *
    * Returns a sequence of Vectors, one for each FeatureSpace included
    * in this pipeline.
    */
  def apply(document: JObject): Vector = {
    if (!isFrozen) throw new IllegalStateException("Pipeline must be primed before use.")
    // apply the feature pipeline to the document
    val computedVectors = applyFeatureGraph(document)
    // get vectors that have been output by the output stage.
    val vec = computedVectors(FeatureGraph.OutputStage).asInstanceOf[Seq[Vector]]
    // Need to create a sparse vector which is a concatenation of all output vectors
    concatenateVectors(vec)
  }

  /**
    * Concatenates the sequence of vectors, transforming their indices appropriately.
    *
    * @param vectors Sequence of vectors to combine. Order should be the same as listed
    *                by the outputTransformNames field.
    * @return
    */
  def concatenateVectors(vectors: Seq[Vector]): Vector = {
    if (vectors.size == 1) {
      /* in many Alloys, we're likely to only have a single transform
       * connected to the output, so we can skip concatenation and
       * avoid the overhead of duplicating */
      val vector = vectors.head
      val expected = outputDimensions.get.head._2
      assert(vector.size == expected, "Incorrect dimensionality")
      vector
    } else {
      val numNonzeroDimensions = vectors.map(_.numActives).sum
      // create underlying sparse data structures
      val newIndexes = new Array[Int](numNonzeroDimensions)
      val newValues = new Array[Double](numNonzeroDimensions)
      var startIndex = 0
      var startOffset = 0
      outputDimensions.get.zip(vectors).foreach {
        case ((_, expectedDimension), vector) => {
          // each vector will have its full dimension so grab that
          val dimension = vector.size
          // assert the vector dimension matches the one we have taken earlier.
          assert(dimension == expectedDimension, "Incorrect dimensionality")
          vector.foreachActive((index, value) => {
            /* Don't need to check that this index is in bounds, because
             * the vector dimension matches what we expect */
            newIndexes(startIndex) = startOffset + index
            newValues(startIndex) = value
            startIndex += 1
          })
          startOffset += dimension
        }
      }
      // create new sparse vector from it all
      Vectors.sparse(startOffset, newIndexes, newValues)
    }
  }

  /** Saves each archivable transforms and generates the total pipeline JSON.
    * We don't need to store expected dimensions since we can recover them from
    * the transformers on load.
    *
    * See {@link com.idibon.ml.common.Archivable}
    */
  def save(writer: Alloy.Writer): Option[JObject] = {
    FeatureGraph.saveGraph(writer, this.state.transforms, this.state.pipeline)
  }

  /**
    * Use this function to enforce feature selection essentially.
    *
    * Feature selection basically means you're pruning what features
    * you're not interested in storing. Since this function does not
    * tell you what features to remove, it is hence called prune.
    *
    * What tells you what to remove, is the passed in predicate function.
    *
    * @param predicate this tells us what feature to remove. It works on the
    *                  global feature space (i.e. the output of apply) and thus
    *                  should expect integers representing those values.
    */
  def prune(predicate: Int => Boolean) {
    if (!isFrozen) throw new IllegalStateException("Pipeline must be primed")

    outputDimensions.get.zip(priorDimensions.get).par.foreach {
      case ((transform, _), priorDimension) => {
        // convert native "prune" indices into global predicate indices
        transform.prune((local: Int) => predicate(priorDimension + local))
      }
    }
  }

  /** Returns the unique feature corresponding to a specific dimension
    *
    * In some cases, some or all indices in the feature vector generated by the
    * FeaturePipeline correspond to specific features in the learned model
    * vocabulary. This method can be used to convert from the abstract Vector
    * representation back to the original Feature objects, to facilitate
    * understanding model performance
    *
    * @param index the dimensional index to invert
    * @return the original feature, or None if inversion is not possible
    */
  def getFeatureByIndex(index: Int): Option[Feature[_]] = {
    if (!isFrozen) throw new IllegalStateException("Pipeline must be primed.")

    outputDimensions.get.zip(priorDimensions.get).find({
      case ((_, dim), priorDim) => index >= priorDim && index < dim + priorDim
    }).flatMap({
      case ((xf, _), priorDim) => xf.getFeatureByIndex(index - priorDim)
    })
  }

  /** Returns unique features corresponding to multiple dimensions
    *
    * This method behaves identically to
    * {@link com.idibon.ml.feature.FeaturePipeline#getFeatureByIndex}, but
    * operates on a list of ascending feature indices. This can provide higher
    * performance than transforming each index individually
    */
  def getFeaturesBySortedIndices(indices: collection.GenTraversableOnce[Int]):
      Seq[Option[Feature[_]]] = {
    if (!isFrozen) throw new IllegalStateException("Pipeline must be primed")

    var transforms = outputDimensions.get.zip(priorDimensions.get)
    var previous = -1
    indices.toIterator.map(current => {
      if (previous > current)
        throw new IllegalArgumentException("Indices are not sorted")

      previous = current
      /* advance transforms to the current TerminableTransformer to invert
       * the current index; all future indices will need either this
       * transform or a later one */
      transforms = transforms.dropWhile({ case ((_, dim), offset) => {
        current >= offset + dim
      }})

      // and convert the current index to its feature, if possible
      transforms.headOption.flatMap({ case ((xf, _), offset) => {
        xf.getFeatureByIndex(current - offset)
      }})
    }).toList
  }

  /** Returns unique features corresponding to multiple dimensions
    *
    * Same as {@link com.idibon.ml.feature.FeaturePipeline#getFeatureByIndex},
    * but returns the Feature for each active index in the Vector
    *
    * @param featureVector a raw feature vector to invert
    */
  def getFeaturesByVector(featureVector: Vector) = featureVector match {
    case sparse: org.apache.spark.mllib.linalg.SparseVector => {
      getFeaturesBySortedIndices(sparse.indices)
    }
    case _ => {
      getFeaturesBySortedIndices(0 until featureVector.size)
    }
  }
}

/** Paired loader class for FeaturePipeline instances */
class FeaturePipelineLoader
    extends ArchiveLoader[FeaturePipeline]
    with FeatureGraphLoader {
  /** Load the FeaturePipeline from the Alloy
    *
    * See {@link com.idibon.ml.feature.FeaturePipeline}
    *
    * @param reader    an Alloy.Reader configured for loading all of the
    *                  resources for this FeaturePipeline
    * @param config    the JSON configuration object generated when this
    *                  pipeline was last saved
    * @return this
    */
  def load(engine: Engine, reader: Option[Alloy.Reader], config: Option[JObject]): FeaturePipeline = {
    FeaturePipeline.logger.trace(s"[$this] loading")

    val transforms = loadTransformers(engine, reader, config)
    val pipeJson = loadPipelineEntries(config)

    val pipeline = FeaturePipeline.bind(transforms, pipeJson)
    reader match {
      case Some(reader) => {
        pipeline.freeze()
      }
      case None => pipeline
    }
  }
}

private[feature] object FeaturePipeline {
  val DefaultName = "<undefined>"

  val logger = Logger(org.slf4j.LoggerFactory
    .getLogger(classOf[FeaturePipeline]))

  /** Creates a bound state graph from the provided input state
    *
    * Helper method for load, also used for programmatic creation of
    * FeaturePipline instances by the FeaturePipelineBuilder
    *
    * @param transforms  all of the transforms and names used in the pipeline
    * @param pipeline    the pipeline structure
    * @return this
    */
  def bind(transforms: Map[String, FeatureTransformer],
           pipeline: Seq[PipelineEntry]): FeaturePipeline = {

    val graph = FeaturePipeline.bindGraph(transforms, pipeline)

    new FeaturePipeline(new LoadState(graph, pipeline, transforms), None)
  }


  /** Validates and generates a bound, callable feature pipeline
    *
    * @param transforms  all of the defined feature transformer objects
    *                    in the feature pipeline
    * @param entries     the unordered list of edges loaded from JSON
    * @return   a transformation function that accepts a document JSON
    *           as input and returns the concatenated set of feature outputs
    */
  def bindGraph(transforms: Map[String, FeatureTransformer],
    entries: Seq[PipelineEntry],
    pipelineName: String = DefaultName):
      FeatureGraph = {

    // verify that all of the named outputs implement TerminableTransformer
    val invalidOutputs = collectOutputs(transforms, entries)
      .find({ case (_, xf) => !xf.isInstanceOf[TerminableTransformer]})

    if (!invalidOutputs.isEmpty) {
      logger.error(s"[$pipelineName/$$output] non-terminable: ${invalidOutputs.map(_._1)}")
      throw new IllegalArgumentException("Non-terminable output transform")
    }

    FeatureGraph[Vector](pipelineName, transforms, entries,
      Seq(FeatureGraph.DocumentInput))
  }

  /* Returns the name and transform of each FeatureTransformer output */
  private[feature] def collectOutputs(
    transforms: Map[String, FeatureTransformer],
    entries: Seq[PipelineEntry]):
      Seq[(String, FeatureTransformer)] = {

    entries.find(_.name == FeatureGraph.OutputStage)
      .get.inputs.map(name => (name, transforms(name)))
  }

}

// all internal state maintained by the FeaturePipeline
private[this] case class LoadState(graph: FeatureGraph,
  pipeline: Seq[PipelineEntry],
  transforms: Map[String, FeatureTransformer])


// Schema for a single entry within the pipeline JSON array
private[feature] case class SinglePipeline(pipeline: Seq[PipelineEntry], transform: Seq[TransformEntry])

// Schema for the array that holds all pipelines within the pipeline JSON array
private[feature] case class Pipelines(pipelines: Seq[SinglePipeline])

// Binds a reified FeatureTransformer within a processing graph
private[feature] case class BoundTransform(name: String,
                                           transform: (AnyMap[String, Any]) => Any)

// An independently-processable stage within the processing graph.
private[feature] case class PipelineStage(killList: List[String],
                                          transforms: List[BoundTransform])
