import scala.collection.mutable.{HashSet => MutableSet}
import scala.collection.{Map => AnyMap}
import scala.reflect.runtime.universe.{MethodMirror, Type, typeOf}

import org.json4s._

import org.apache.spark.mllib.linalg.Vector
import com.typesafe.scalalogging.StrictLogging

import com.idibon.ml.alloy.Alloy
import com.idibon.ml.common.Reflect._

package com.idibon.ml.feature {

  /** The feature pipeline transforms documents into feature vectors
    */
  class FeaturePipeline extends Archivable {

    /** Applies the entire feature pipeline to the provided document.
      *
      * Returns a sequence of Vectors, one for each FeatureSpace included
      * in this pipeline.
      */
    def apply(document: JObject): Seq[Vector] = {
      _state.map(s => {
        /* the intermediate data passed between pipeline stages; initialize
         * with the document under the reserved name "$document" */
        val intermediates = scala.collection.mutable.Map[String, Any](
          FeaturePipeline.DocumentInput -> document
        )
        for (stage <- s.graph) {
          intermediates ++= stage.transforms
            .map(xf => (xf.name -> xf.transform(intermediates)))

          intermediates --= stage.killList
        }

        intermediates(FeaturePipeline.OutputStage).asInstanceOf[Seq[Vector]]
      }).getOrElse(throw new IllegalStateException("Pipeline not loaded"))
    }

    /** Saves each archivable transforms and generates the total pipeline JSON
      *
      * See {@link com.idibon.ml.feature.Archivable}
      */
    def save(writer: Alloy.Writer): Option[JObject] = {
      // todo: store dimensions of output vectors here?
      _state.map(s =>
        JObject(List(
          JField("transforms", JArray(s.transforms.map({ case (name, xf) => {
            // archive each Archivable transform
            val configJson = xf match {
              case archive: Archivable => archive.save(writer.within(name))
              case _ => None
            }
            // and map to the entry in the transforms array
            JObject(List(
              JField("name", JString(name)),
              JField("class", JString(xf.getClass.getName)),
              JField("config", configJson.getOrElse(JNothing))
            ))
          }}).toList)),
          JField("pipeline", JArray(s.pipeline.map(pipe => {
            // construct an entry in the "pipeline" array for each PipelineEntry
            JObject(List(
              JField("name", JString(pipe.name)),
              JField("inputs", JArray(pipe.inputs.map(i => JString(i))))
            ))
          }).toList))
        )))
    }

    @volatile private var _state: Option[LoadState] = None

    /** Load the FeaturePipeline from the Alloy
      *
      * See {@link com.idibon.ml.feature.FeaturePipeline}
      *
      * @param reader    an Alloy.Reader configured for loading all of the
      *   resources for this FeaturePipeline
      * @param config    the JSON configuration object generated when this
      *   pipeline was last saved
      */
    def load(reader: Alloy.Reader, config: Option[JObject]): this.type = {
      // configure format converters for json4s
      implicit val formats = DefaultFormats

      val xfJson = (config.get \ "transforms").extract[List[TransformEntry]]
      val pipeJson = (config.get \ "pipeline").extract[List[PipelineEntry]]

      /* instantiate all of the transformers and pass any configuration data
       * to them, generating a map of the transform name to the reified
       * transformer object. */
      val transforms = xfJson.map(obj => {
        (obj.name -> reify(reader.within(obj.name), obj))
      }).toMap

      val graph = FeaturePipeline.bindGraph(transforms, pipeJson)

      /* grab all of the feature transformers bound to the output stage
       * these are the possible sources for significant feature inversion */
      val outputs = pipeJson.find(_.name == "$output")
        .map(_.inputs.map(i => transforms(i))).getOrElse(List.empty)

      _state = Some(new LoadState(graph, pipeJson, outputs, transforms))
      this
    }

    /** Reifies a single FeatureTransformer
      *
      * Instantiates and, for Archivable objects, loads, the FeatureTransformer
      * represented by transformDef.
      *
      * @param reader        an Alloy.Reader configured for loading all of the
      *   resources for this FeatureTransformer
      * @param transformDef  the instance information for this transform -
      *   name, class and optional configuration information.
      * @return   tuple of the transformer name and the reified object
      */
    private def reify(reader: Alloy.Reader, entry: TransformEntry):
        FeatureTransformer = {

      val transform = Class.forName(entry.`class`)
        .newInstance.asInstanceOf[FeatureTransformer]

      transform match {
        // load and return the transform from the Alloy, if it's Archivable
        case archived: FeatureTransformer with Archivable => {
          // grab the configuration data for this transformer, if any
          archived.load(reader, entry.config)
        }
        // otherwise just return the transform
        case _ => transform
      }
    }
  }

  private[feature] object FeaturePipeline extends StrictLogging {
    val OutputStage = "$output"
    val DocumentInput = "$document"

    /** Validates and generates a bound, callable feature pipeline
      *
      * @param transforms  all of the defined feature transformer objects
      *   in the feature pipeline
      * @param entries     the unordered list of edges loaded from JSON
      * @return   a transformation function that accepts a document JSON
      *   as input and returns the concatenated set of feature outputs
      */
    private[feature] def bindGraph(transforms: Map[String, FeatureTransformer],
        entries: Seq[PipelineEntry]): Seq[PipelineStage] = {

      // create a map of transform name to required inputs
      val dependency = entries.map(obj => (obj.name -> obj.inputs)).toMap

      /* generate a map from every named transformer to the apply method for
       * that transformer; throw an exception if the method isn't defined */
      val applyMirrors: Map[String, MethodMirror] =
        transforms.map({ case (name, transformer) => {
          (name -> getMethod(transformer, "apply")
            .getOrElse(throw new IllegalArgumentException("No apply:" + name)))
        }}).toMap

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
      (sortedDependencies zip futureNeeds).map(
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
                if (inputNames.isEmpty)
                  throw new IllegalArgumentException("No output from pipeline")

                /* all of the stages feeding into the $output stage should
                 * return vectors. log a warning message if not. */
                (inputNames zip inputTypes).filterNot(_._2 <:< typeOf[Vector])
                  .foreach(i => {
                    logger.warn(s"Output of ${i._1} may not be a Vector")
                  })

                /* and return the binding function that just pivots the input
                 * data from the intermediates map into a sequence */
                (intermediates: AnyMap[String, Any]) => {
                  inputNames.map(intermediates(_))
                }
              }
              /* for proper transform stages, return a thunk that deserializes
               * the named inputs from the intermediates map, then calls the
               * reflected apply method to generate the output */
              case _ => {
                val reflected = applyMirrors(current)
                if (!isValidInvocation(reflected.symbol, List(inputTypes)))
                  logger.warn(s"Binding for $current may not satisfy type signature")

                /* and return a thunk that calls the apply method using the
                 * input values from the intermediates map.
                 *
                 * TODO: handle variadic arguments
                 */
                (intermediates: AnyMap[String, Any]) => {
                  reflected(inputNames.map(intermediates(_)): _*)
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
        }})
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
        (entry.name -> (entry.inputs, MutableSet[String]()))
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
                throw new IllegalArgumentException("Cyclic graph: " +name)
              Some(transitive)
            }
          }) match {
            // return an immutable copy of the node's transitive dependencies
            case Some(x) => x.toSet
            // base case handling for initial document input
            case None if name == FeaturePipeline.DocumentInput => Set()
            // or throw an exception if the node doesn't exist
            case _ => throw new IllegalArgumentException("Missing: " + name)
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
}
