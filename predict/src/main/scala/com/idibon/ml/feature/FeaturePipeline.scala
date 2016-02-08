package com.idibon.ml.feature

import com.typesafe.scalalogging.{StrictLogging, Logger}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.json4s._

import scala.Boolean
import scala.collection.mutable.{HashSet => MutableSet, ListBuffer}
import scala.collection.{Map => AnyMap}
import scala.reflect.runtime.universe.{MethodMirror, typeOf}
import scala.collection.mutable

import com.idibon.ml.alloy.Alloy
import com.idibon.ml.common.{ArchiveLoader, Archivable, Engine}
import com.idibon.ml.common.Reflect._


/** The feature pipeline transforms documents into feature vectors
  */
class FeaturePipeline(
  state: LoadState,
  outputDimensions: Option[Seq[(TerminableTransformer, Int)]] = None)
    extends Archivable[FeaturePipeline, FeaturePipelineLoader]
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
    for (document <- documents) {
      applyFeatureGraph(document)
    }
    freezePipeline()
  }

  /**
    * Helper method to effectively freeze the state of the pipeline.
    *
    * It saves the outputStage to a list, as well as the total dimensions and
    * the size of each outputStage's dimensions.
    */
  private[feature] def freezePipeline(): FeaturePipeline = {
    val terminables = FeaturePipeline
      .collectOutputs(this.state.transforms, this.state.pipeline)
      .map({ case (_, xf) => xf.asInstanceOf[TerminableTransformer] })

    terminables.foreach(_.freeze)

    new FeaturePipeline(this.state,
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
    val intermediates = scala.collection.mutable.Map[String, Any](
      FeaturePipeline.DocumentInput -> document
    )
    for (stage <- this.state.graph) {
      intermediates ++= stage.transforms
        .map(xf => (xf.name -> xf.transform(intermediates)))

      intermediates --= stage.killList
    }
    intermediates
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
    val vec = computedVectors(FeaturePipeline.OutputStage).asInstanceOf[Seq[Vector]]
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
    Some(JObject(List(
      JField("version", JString(FeaturePipeline.SchemaVersion)),
      JField("transforms",
        JArray(this.state.transforms.map({ case (name, xf) => {
          // create the serialized TransformEntry representation
          JObject(List(
            JField("name", JString(name)),
            JField("class", JString(xf.getClass.getName)),
            JField("config",
              Archivable.save(xf, writer.within(name)).getOrElse(JNothing))
          ))
        }
        }).toList)),
      JField("pipeline",
        JArray(this.state.pipeline.map(pipe => {
          // construct an entry in the "pipeline" array for each PipelineEntry
          JObject(List(
            JField("name", JString(pipe.name)),
            JField("inputs", JArray(pipe.inputs.map(i => JString(i))))
          ))
        }).toList))
    )))
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
class FeaturePipelineLoader extends ArchiveLoader[FeaturePipeline] {
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
    // configure format converters for json4s
    implicit val formats = DefaultFormats

    FeaturePipeline.logger.trace(s"[$this] loading")

    val xfJson = (config.get \ "transforms").extract[List[TransformEntry]]
    val pipeJson = (config.get \ "pipeline").extract[List[PipelineEntry]]


    /* instantiate all of the transformers and pass any configuration data
             * to them, generating a map of the transform name to the reified
             * transformer object. */
    val transforms = xfJson.map(obj => {
      FeaturePipeline.checkTransformerName(obj.name)
      val resourceReader = if (reader.isDefined) Some(reader.get.within(obj.name)) else None
      (obj.name -> reify(engine, resourceReader, obj))
    }).toMap
    val pipeline = FeaturePipeline.bind(transforms, pipeJson)
    reader match {
      case Some(reader) => {
        pipeline.freezePipeline()
      }
      case None => pipeline
    }

  }


  /** Reifies a single FeatureTransformer
    *
    * Instantiates and, for Archivable objects, loads, the FeatureTransformer
    * represented by transformDef.
    *
    * @param reader        an Alloy.Reader configured for loading all of the
    *                      resources for this FeatureTransformer
    * @param entry  the instance information for this transform -
    *               name, class and optional configuration information.
    * @return   tuple of the transformer name and the reified object
    */
  private def reify(engine: Engine, reader: Option[Alloy.Reader], entry: TransformEntry):
  FeatureTransformer = {

    val transformClass = Class.forName(entry.`class`)
    ArchiveLoader
      .reify[FeatureTransformer](transformClass, engine, reader, entry.config)
      .getOrElse(transformClass.newInstance.asInstanceOf[FeatureTransformer])
  }
}

private[feature] object FeaturePipeline {
  val OutputStage = "$output"
  val DocumentInput = "$document"
  val DefaultName = "<undefined>"
  val SchemaVersion = "0.0.1"

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

    /* grab all of the feature transformers bound to the output stage
     * these are the possible sources for significant feature inversion */
    val outputs = pipeline.find(_.name == "$output")
      .map(_.inputs.map(i => transforms(i))).getOrElse(List.empty)

    new FeaturePipeline(new LoadState(graph, pipeline, outputs, transforms), None)
  }


  /** Tests if a user-provided name uses a reserved sequence.
    *
    * @param name string to check
    * @return true if the name uses reserved characters, otherwise false
    */
  def checkTransformerName(name: String, pipelineName: String = DefaultName) {
    if (name.charAt(0) == '$')
      logger.warn(s"[$pipelineName/$name] - using reserved name")
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
                pipelineName: String = DefaultName): Seq[PipelineStage] = {

    // create a map of transform name to required inputs
    val dependency = entries.map(obj => (obj.name -> obj.inputs)).toMap

    // an output stage must be defined
    if (entries.find(_.name == FeaturePipeline.OutputStage).isEmpty) {
      logger.error(s"[$pipelineName/$$output]: $$output missing")
      throw new NoSuchElementException("No $output")
    }

    /* generate a map from every named transformer to the apply method for
     * that transformer; throw an exception if the method isn't defined */
    val applyMirrors: Map[String, MethodMirror] =
      transforms.map({ case (name, transformer) => {
        (name -> getMethod(transformer, "apply")
          .getOrElse(throw new IllegalArgumentException(s"No apply: $name")))
      }
      }).toMap

    /* sort all of the individual pipeline entries into a dependency
     * graph, where each entry in the graph is all of the transforms that
     * can be performed at that stage. */
    val sortedDependencies = sortDependencies(entries)

    /* for each dependency stage, compute the set of intermediate values
     * that are needed by future stages; this will be used to form the
     * list of intermediate values that are removed from the intermediate
     * map after executing each pipeline stage. */

    val futureNeeds = scala.collection.mutable.MutableList[Set[String]]()

    /* build the list of future needs in reverse by accumulating all of
     * input references in later stages, and prepending each accumulated
     * set to the list of futureNeeds */
    for (stage <- sortedDependencies.reverse) {
      (futureNeeds.headOption.getOrElse(Set.empty) ++ stage) +=: futureNeeds
    }

    /* track all of the live intermediate values at each stage (initially
     * seeded by the document itself). this will be compared against the
     * future needs to produce a set of intermediate values that should
     * be killed at each processing stage. */
    val liveList = MutableSet(FeaturePipeline.DocumentInput)

    // convert the dependency graph into PipelineStage instances
    val boundGraph = (sortedDependencies zip futureNeeds).map(
      { case (transformNamesInStage, needsAfterStage) => {
        /* create BoundTransform objects for each transform in this stage.
         * the method in the BoundTransform is just a thunk that deserializes
         * the input data from a shared map of live intermediate data, calls
         * the apply() method on the FeatureTrasformer (using reflection),
         * then stores the results of the transform back in the shared map for
         future stages. */
        val bindings = transformNamesInStage.map(current => {

          /* lookup the named input dependencies for this transform, or
           * return an empty list if (for some reason) the transform has
           * no inputs. */
          val inputNames = dependency.getOrElse(current, List())

          /* map all of the named inputs to the actual data types that are
           * expected to be used. inputs named "$input" will just take a JSON
           * object of the document. */
          val inputTypes = inputNames.map(_ match {
            case FeaturePipeline.DocumentInput => typeOf[JObject]
            case otherXformer => applyMirrors(otherXformer).symbol.returnType
          })

          val bindFunction = current match {
            /* for "$output", return a function that concatenates the named
             * inputs into a Seq */
            case FeaturePipeline.OutputStage => {
              /* at least one input to the $output stage must exist for the
               * pipeline to validate. */
              if (inputNames.isEmpty) {
                logger.error(s"[$pipelineName/$$output] no outputs defined")
                throw new IllegalArgumentException("No pipeline output")
              }

              /* all of the stages feeding into the $output stage should
               * return vectors. log a warning message if not. */
              (inputNames zip inputTypes).filterNot(_._2 <:< typeOf[Vector])
                .foreach(i => {
                  logger.warn(s"[$pipelineName/${i._1}] possible non-Vector output")
                })

              /* and return the binding function that just pivots the input
               * data from the intermediates map into a sequence */
              (intermediates: AnyMap[String, Any]) => {
                inputNames.map(n => intermediates(n))
              }
            }
            /* for proper transform stages, return a thunk that deserializes
             * the named inputs from the intermediates map, then calls the
             * reflected apply method to generate the output */
            case _ => {
              val reflected = applyMirrors.get(current).getOrElse({
                logger.error(s"[$pipelineName/$current] no transformer defined")
                throw new NoSuchElementException(current)
              })
              val method = reflected.symbol
              if (!isValidInvocation(method, List(inputTypes)))
                logger.error(s"[$pipelineName/$current] inputs may not satisfy call")

              /* if the FeatureTransformer is declared using multiple
               * parameter lists (for, e.g., function currying), throw
               * an UnsupportedOperationException. */
              if (method.paramLists.size > 1) {
                logger.error(s"[$pipelineName/$current] curried arguments")
                throw new UnsupportedOperationException("Curried arguments")
              }

              /* if the method accepts variadic arguments for the last
               * parameter, the thunk will need to convert:
               * (A, B, C, D, ...) into (A, B, C, D, List(...)).
               * cache the number of non-variadic arguments in the
               * parameter list */
              val variadicIndex = getVariadicParameterType(method,
                method.paramLists.headOption.getOrElse(List.empty))
                .map(_ => method.paramLists.head.length - 1)

              /* and return a thunk that calls the apply method using the
               * input values from the intermediates map. */
              (intermediates: AnyMap[String, Any]) => {
                // pivot the intermediates table into an argument list
                val args = inputNames.map(n => intermediates(n))
                /* if the apply method is variadic, slice off the tail
                 * (variadic) arguments, and append the resulting list as a
                 * single element (i.e., do not concatenate the lists).
                 * otherwise, just call the reflected method with the
                 * unmodified argumnets list */
                variadicIndex.map(i => {
                  val va = args.slice(0, i - 1) :+ args.slice(i - 1, args.size)
                  reflected(va: _*)
                }).getOrElse(reflected(args: _*))
              }
            }
          }

          // return a BoundTransform for the bind function
          new BoundTransform(current, bindFunction)
        })

        // anything that is alive but not needed in the future is killable
        val killable = liveList & needsAfterStage
        liveList --= killable

        // add all of the transforms from this stage as live intermediate values
        liveList ++= transformNamesInStage

        // yield the final pipeline stage
        new PipelineStage(killable.toList, bindings)
      }
      })


    // verify that every output transform implements TerminableTransformer
    val invalidOutputs = collectOutputs(transforms, entries)
      .find({ case (_, xf) => !xf.isInstanceOf[TerminableTransformer]})

    if (!invalidOutputs.isEmpty) {
      logger.error(s"[$pipelineName/$$output] non-terminable: ${invalidOutputs.map(_._1)}")
      throw new IllegalArgumentException("Non-terminable output transform")
    }

    boundGraph
  }

  /* Returns the name and transform of each FeatureTransformer output */
  private[feature] def collectOutputs(
    transforms: Map[String, FeatureTransformer],
    entries: Seq[PipelineEntry]): Seq[(String, FeatureTransformer)] = {

    entries.find(_.name == FeaturePipeline.OutputStage)
      .get.inputs.map(name => (name, transforms(name)))
  }

  /** Orders the transformers in a pipeline in dependency order
    *
    * Orders all of the entries in a pipeline based on the order that
    * the transform must be applied to satisfy the defined input
    * dependencies, then splits the sorted list into variable-sized
    * stages of independent processing (i.e., any entry in stage N
    * depends only on the results of transforms in [Stage 0..N-1])
    */
  private[feature] def sortDependencies(pipeline: Seq[PipelineEntry]):
      List[List[String]] = {

    /* recursively construct the full dependency graph using the partial
     * results of previously-calculated dependencies. results in a table
     * (name -> (List(direct dependencies), Set(all dependencies)) */
    val dependencies = pipeline.map(entry => {
      (entry.name ->(entry.inputs, MutableSet[String]()))
    }).toMap

    /* recursive call to determine all transitive dependencies in
     * pipeline entry "name." returns the full dependency list */
    def getDeps(name: String, history: MutableSet[String]): Set[String] = {
      if (history.contains(name)) {
        /* the graph is already processed for this node, so just return
         * an immutable copy of the previously-calculated results */
        dependencies(name)._2.toSet
      } else {
        // calculate transitive dependencies for this entry, if it exists
        dependencies.get(name).flatMap({
          case (named: List[String], transitive: MutableSet[String]) => {
            /* add all of this node's direct dependencies to the set of
             * all transitive dependencies */
            transitive ++= named
            /* add the current object to the history list, to prevent
             * infinite recursion in case of cycles */
            history += name
            /* add all of the computed dependencies for each direct
             * dependency to the dependency list */
            val indirect = named.map(n => getDeps(n, history))
              .foldLeft(MutableSet[String]())(_ ++= _)
            transitive ++= indirect
            // nodes should never appear as their own dependencies
            if (transitive.contains(name))
              throw new IllegalArgumentException("Cyclic graph: " + name)
            Some(transitive)
          }
        }) match {
          // return an immutable copy of the node's transitive dependencies
          case Some(x) => x.toSet
          // base case handling for initial document input
          case None if name == FeaturePipeline.DocumentInput => Set()
          // or throw an exception if the node doesn't exist
          case _ => throw new NoSuchElementException(name)
        }
      }
    }

    val graphedEntries = MutableSet[String]()
    pipeline.foreach(entry => getDeps(entry.name, graphedEntries))

    /* sort the pipeline stages based on the order that they must be
     * processed to fulfill the dependency graph: returns true if A
     * is a dependency of B, or if A has fewer total dependencies than B. */
    val sortedEntries = graphedEntries.toList.sortWith((a, b) => {
      dependencies(b)._2.contains(a) ||
        dependencies(a)._2.size <= dependencies(b)._2.size
    })

    val stages = scala.collection.mutable.MutableList[List[String]]()
    val currentStage = MutableSet[String]()

    // split the sorted pipeline entries into independent pipeline stages
    sortedEntries.foreach(entry => {
      if (dependencies(entry)._1.exists(currentStage.contains(_))) {
        /* if any of the named dependencies for the transform exist
         * within the current stage, then a new stage must be created
         * for the transform */
        stages += currentStage.toList
        currentStage.clear
      }
      currentStage += entry
    })
    // add the final stage (should be $output, in well-formed graphs)
    if (!currentStage.isEmpty) stages += currentStage.toList
    stages.toList
  }
}

// all internal state maintained by the FeaturePipeline
private[this] case class LoadState(graph: Seq[PipelineStage],
                                      pipeline: Seq[PipelineEntry],
                                      inverters: Seq[FeatureTransformer],
                                      transforms: Map[String, FeatureTransformer])


// Schema for each entry within the transforms JSON array
private[feature] case class TransformEntry(name: String, `class`: String,
                                           config: Option[JObject])

// Schema for each entry within the pipeline JSON array
private[feature] case class PipelineEntry(name: String, inputs: List[String])

// Binds a reified FeatureTransformer within a processing graph
private[feature] case class BoundTransform(name: String,
                                           transform: (AnyMap[String, Any]) => Any)

// An independently-processable stage within the processing graph.
private[feature] case class PipelineStage(killList: List[String],
                                          transforms: List[BoundTransform])
