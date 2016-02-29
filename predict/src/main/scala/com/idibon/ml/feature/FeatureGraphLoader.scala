package com.idibon.ml.feature

import com.idibon.ml.alloy.Alloy
import com.idibon.ml.common.{Engine, ArchiveLoader}

import org.json4s.JObject

/** Mixin for loader classes which load a feature graph from an Alloy
  */
trait FeatureGraphLoader {

  /** Loads and reifies all of the individual transforms in the graph
    *
    * @param engine current engine context
    * @param reader interface to read persisted data from the alloy
    * @param config saved feature graph configuration JSON
    * @return a map from transform names to reified feature transformers
    */
  def loadTransformers(engine: Engine, reader: Option[Alloy.Reader],
    config: Option[JObject]): Map[String, FeatureTransformer] = {

    implicit val formats = org.json4s.DefaultFormats

    (config.get \ "version").extract[Option[String]] match {
      /* for backwards compatibility with some config files, treat a
       * pipeline with no version the same as the initial version, 0.0.1 */
      case Some(FeatureGraph.SchemaVersion_001) | None => {
        val entries = (config.get \ "transforms").extract[List[TransformEntry]]
        entries.map(e => {
          // verify that the transform name is not reserved
          if (e.name.charAt(0) == '$')
            throw new IllegalArgumentException(s"reserved name: ${e.name}")

          /* use the paired archive loader for archivable transforms, or
           * instantiate a transform if the class is not archivable */
          val klass = Class.forName(e.`class`)
          (e.name -> ArchiveLoader
            .reify[FeatureTransformer](klass, engine, reader.map(_.within(e.name)), e.config)
            .getOrElse(klass.newInstance.asInstanceOf[FeatureTransformer]))
        }).toMap
      }
      case Some(other) => {
        throw new UnsupportedOperationException(s"Unknown version: $other")
      }
    }
  }

  /** Loads the list of pipeline dependencies
    *
    * @param config saved feature graph configuration JSON
    * @return list of per-transformer dependencies
    */
  def loadPipelineEntries(config: Option[JObject]): Seq[PipelineEntry] = {
    implicit val formats = org.json4s.DefaultFormats

    (config.get \ "version").extract[Option[String]] match {
      /* for backwards compatibility with some config files, treat a
       * pipeline with no version the same as the initial version, 0.0.1 */
      case Some(FeatureGraph.SchemaVersion_001) | None => {
        (config.get \ "pipeline").extract[List[PipelineEntry]]
      }
      case Some(other) => {
        throw new UnsupportedOperationException(s"Unknown version: $other")
      }
    }
  }
}

// == JSON Schemas ==

/** JSON schema for a single FeatureTransformer */
private[feature] case class TransformEntry(name: String, `class`: String,
                                           config: Option[JObject])

/** JSON schema for an edge between FeatureTransformers */
private[feature] case class PipelineEntry(name: String, inputs: List[String])
