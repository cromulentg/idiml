package com.idibon.ml.feature

import scala.collection.mutable
import scala.reflect.runtime.universe.{MethodMirror, Type, typeOf, TypeTag}

import com.idibon.ml.common.Reflect._
import com.idibon.ml.common.{Archivable, Engine}
import com.idibon.ml.alloy.Alloy

import org.json4s._

/** Bound, callable graph of feature transformers
  *
  * @param name name for the graph
  * @param graph list of dependent feature transforms to apply
  */
private[feature] class FeatureGraph(val name: String,
  private[feature] val graph: Seq[PipelineStage]) {

  /** Transforms a set of input against the entire FeatureGraph
    *
    * Operations are applied directly to the input (mutable) Map, and the
    * original input data may be eliminated during the application of the
    * graph.
    *
    * Returns the resulting Map after applying all of the pipeline stages
    * in order, storing the intermediate results for each transform in
    * a key using that transform's name.
    *
    * @param input initial set of data for processing
    * @return the transformed map
    */
  def apply(input: mutable.Map[String, Any]): mutable.Map[String, Any] = {
    graph.foreach(stage => {
      // add all of the intermediate results for the current pipeline stage
      input ++= stage.transforms.map(xf => (xf.name -> xf.transform(input)))
      // evict all of the items that won't be referenced again from the map
      input --= stage.killList
    })
    input
  }
}

/** Binds a set of named feature transforms (nodes) and a list of per-transform
  * inputs (edges) into a FeatureGraph.
  */
private[feature] object FeatureGraph {

  /** Schema used for feature graphs in initial release. */
  val SchemaVersion_001 = "0.0.1"
  /** Current version of the graph JSON and alloy schema */
  val SchemaVersion = SchemaVersion_001
  /** Stage name for the results of the graph transformation */
  val OutputStage = "$output"
  /** Reserved name for the input document, must be of type JObject */
  val DocumentInput = "$document"


  // borrow the FeaturePipeline's logger, rather than creating yet another
  private [this] val logger = FeaturePipeline.logger

  /** Generates a thunk function for reflectively calling feature transforms
    *
    * Each feature transform's apply method is defined directly using
    * transform-appropriate data types (e.g., Feature[String], Seq[Token]),
    * but the intermediate values for the graph are stored in a
    * mutable.Map[String, Any]. The thunk function must deserialize the
    * correct set of inputs from the map, then invoke the apply method.
    *
    * Raises errors if the apply method signature is invalid
    *
    * @tparam T the desired output type for the graph (e.g., Vector)
    * @param graphName a name identifying the graph
    * @param transformName the name of the pipeline entry
    * @param applyMethod the transform's apply method, if present
    * @param inputNames the named input arguments for the transform
    * @param inputTypes the data types identified (through reflection) of the
    *   return values of the transforms referenced in inputNames
    * @return a thunk function for the FeatureTransformer
    */
  private[this] def thunk[T: TypeTag](graphName: String,
    transformName: String,
    applyMethod: Option[MethodMirror],
    inputNames: Seq[String],
    inputTypes: Seq[Type]):
      Function1[collection.Map[String, Any], Any] = transformName match {

    /* for "$output", return a function that concatenates the named
     * inputs into a Seq */
    case OutputStage => {
      /* at least one input to the $output stage must exist for the
       * pipeline to validate. */
      if (inputNames.isEmpty) {
        logger.error(s"[$graphName/$$output] no outputs defined")
        throw new IllegalArgumentException("No pipeline output")
      }

      /* all of the stages feeding into the $output stage should
       * return the output type. log a warning message if not. */
      (inputNames zip inputTypes).filterNot(_._2 <:< typeOf[T])
        .foreach(i => {
          logger.error(s"[$graphName/${i._1}] - possible invalid output")
        })

      /* and return the binding function that just pivots the input
       * data from the intermediates map into a Seq[T] */
      (intermediates: collection.Map[String, Any]) => {
        inputNames.map(n => intermediates(n))
      }
    }
    /* for proper transform stages, return a thunk that deserializes
     * the named inputs from the intermediates map, then calls the
     * reflected apply method to generate the output */
    case _ => {
      val reflected = applyMethod.getOrElse({
        logger.error(s"[$graphName/$transformName] no transformer defined")
        throw new NoSuchElementException(transformName)
      })
      val method = reflected.symbol
      if (!isValidInvocation(method, List(inputTypes.toList)))
        logger.error(s"[$graphName/$transformName] inputs may not satisfy call")

      /* if the FeatureTransformer is declared using multiple
       * parameter lists (for, e.g., function currying), throw
       * an UnsupportedOperationException. */
      if (method.paramLists.size > 1) {
        logger.error(s"[$graphName/$transformName] curried arguments")
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
      (intermediates: collection.Map[String, Any]) => {
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

  /** Creates a FeatureGraph from a set of named transforms and their inputs
    *
    * @tparam T the expected output type of the feature graph
    * @param graphName the name of the feature graph
    * @param transforms all of the named transforms in the graph
    * @param entries unordered list of dependencies for each transform
    */
  def apply[T: TypeTag](graphName: String,
    transforms: Map[String, FeatureTransformer],
    entries: Seq[PipelineEntry]):
      FeatureGraph = {

    // create a map of transform name to required inputs
    val dependency = entries.map(obj => (obj.name -> obj.inputs)).toMap

    // an output stage must be defined
    if (entries.find(_.name == OutputStage).isEmpty) {
      logger.error(s"[$graphName/$$output]: $$output missing")
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

    val futureNeeds = mutable.MutableList[Set[String]]()

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
    val liveList = mutable.Set(DocumentInput)

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
            case DocumentInput => typeOf[JObject]
            case otherXformer => applyMirrors(otherXformer).symbol.returnType
          })

          val bindFunction = thunk[T](graphName, current,
            applyMirrors.get(current), inputNames, inputTypes)

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


    new FeatureGraph(graphName, boundGraph)
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
      (entry.name ->(entry.inputs, mutable.Set[String]()))
    }).toMap

    /* recursive call to determine all transitive dependencies in
     * pipeline entry "name." returns the full dependency list */
    def getDeps(name: String, history: mutable.Set[String]): Set[String] = {
      if (history.contains(name)) {
        /* the graph is already processed for this node, so just return
         * an immutable copy of the previously-calculated results */
        dependencies(name)._2.toSet
      } else {
        // calculate transitive dependencies for this entry, if it exists
        dependencies.get(name).flatMap({
          case (named: List[String], transitive: mutable.Set[String]) => {
            /* add all of this node's direct dependencies to the set of
             * all transitive dependencies */
            transitive ++= named
            /* add the current object to the history list, to prevent
             * infinite recursion in case of cycles */
            history += name
            /* add all of the computed dependencies for each direct
             * dependency to the dependency list */
            val indirect = named.map(n => getDeps(n, history))
              .foldLeft(mutable.Set[String]())(_ ++= _)
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
          case None if name == DocumentInput => Set()
          // or throw an exception if the node doesn't exist
          case _ => throw new NoSuchElementException(name)
        }
      }
    }

    val graphedEntries = mutable.Set[String]()
    pipeline.foreach(entry => getDeps(entry.name, graphedEntries))

    /* sort the pipeline stages based on the order that they must be
     * processed to fulfill the dependency graph: returns true if A
     * is a dependency of B, or if A has fewer total dependencies than B. */
    val sortedEntries = graphedEntries.toList.sortWith((a, b) => {
      dependencies(b)._2.contains(a) ||
        dependencies(a)._2.size <= dependencies(b)._2.size
    })

    val stages = mutable.MutableList[List[String]]()
    val currentStage = mutable.Set[String]()

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

  /** Saves a feature graph to an alloy using the default schema
    *
    * @param writer writer interface for this graph's namespace
    * @param transforms all of the transforms in the feature graph
    * @param pipeline the edges connecting transforms
    */
  private[feature] def saveGraph(writer: Alloy.Writer,
    transforms: Map[String, FeatureTransformer],
    pipeline: Seq[PipelineEntry]):
      Option[JObject] = {
    Some(JObject(List(
      JField("version", JString(SchemaVersion)),
      JField("transforms",
        JArray(transforms.map({ case (name, xf) => {
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
        JArray(pipeline.map(pipe => {
          // construct an entry in the "pipeline" array for each PipelineEntry
          JObject(List(
            JField("name", JString(pipe.name)),
            JField("inputs", JArray(pipe.inputs.map(i => JString(i))))
          ))
        }).toList))
    )))
  }
}
