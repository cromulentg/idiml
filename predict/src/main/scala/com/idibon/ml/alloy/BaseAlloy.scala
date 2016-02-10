package com.idibon.ml.alloy

import scala.collection.JavaConverters._
import java.util.{Date, TimeZone}

import com.idibon.ml.predict._
import com.idibon.ml.common.{Archivable, ArchiveLoader, Engine}
import com.idibon.ml.feature._

import org.json4s._
import org.json4s.JsonDSL._
import org.json4s.native.JsonMethods
import com.typesafe.scalalogging.StrictLogging

/** Alloy storing models and meta-data in common locations
  *
  * @param labels all labels assigned by any model in the Alloy
  * @param models top-level models
  */
case class BaseAlloy[T <: PredictResult with Buildable[T, Builder[T]]](
  name: String, labels: Seq[Label], models: Map[String, PredictModel[T]])
    extends Alloy[T] with StrictLogging {

  // cache the mapping from UUID strings to the label
  private[this] val _uuidToLabel = labels.map(l => l.uuid.toString -> l).toMap

  /** Returns the Label with a specific UUID
    *
    * @param uuid
    */
  def translateUUID(uuid: String) = _uuidToLabel.get(uuid).getOrElse(null)

  /** Processes a document using all top-level predictive models
    *
    * @param json document to analyze, represented as a JSON AST
    * @param options optional configuration for the returned results
    */
  def predict(json: JObject, options: PredictOptions) = {
    val doc = Document.document(json)
    models.flatMap({ case (_, m) => m.predict(doc, options)}).toList.asJava
  }

  /** Returns all of the labels in the Alloy */
  def getLabels = labels.toList.asJava

  /** Saves the alloy to a persistent device
    *
    * @param writer alloy writer
    */
  def save(writer: Alloy.Writer) {
    val spec = BaseAlloy.saveManifest(writer, this)
    logger.info(s"save $name, version ${spec.VERSION}")
    spec.saveModels(writer, this.models)
    spec.saveLabels(writer, this.labels)

    // save off all of the optional data possibly included by the furnace
    if (this.isInstanceOf[HasValidationData]) {
      HasValidationData.save(writer,
        this.asInstanceOf[BaseAlloy[T] with HasValidationData])
    }

    if (this.isInstanceOf[HasTrainingConfig]) {
      HasTrainingConfig.save(writer,
        this.asInstanceOf[Alloy[T] with HasTrainingConfig])
    }
  }
}

/** Generic loader for BaseAlloy derived alloys */
object BaseAlloy extends StrictLogging {

  private[alloy] val CURRENT_SPEC: BaseAlloySpecVersion = BaseAlloy_1
  private[this] val MANIFEST_JSON = "manifest.json"

  /** Alternate constructor for reifying from a persistent source
    *
    * @param engine Engine state to use for reified models
    * @param reader Alloy reader
    */
  def load[T <: PredictResult with Buildable[T, Builder[T]]](
      engine: Engine, reader: Alloy.Reader): BaseAlloy[T] = {
    implicit val formats = org.json4s.DefaultFormats

    val man = loadManifest(reader)

    val spec = man.specVersion match {
      case BaseAlloy_1.VERSION => BaseAlloy_1
      case _ => throw new UnsupportedOperationException(s"${man.specVersion}")
    }

    logger.info(s"Load ${man.name}, version ${man.specVersion}, ${man.createdAt}")
    new BaseAlloy(man.name, spec.loadLabels(reader),
      spec.loadModels[T](engine, reader))
  }

  /** Loads an Alloy manifest from a Reader
    *
    * @param reader Alloy reader
    */
  def loadManifest(reader: Alloy.Reader): BaseAlloyManifest = {
    implicit val formats = org.json4s.DefaultFormats

    val resource = reader.resource(MANIFEST_JSON)
    try {
      JsonMethods.parse(Codec.String.read(resource))
        .extract[BaseAlloyManifest]
    } finally {
      resource.close
    }
  }

  /** Generates and saves the Alloy manifest
    *
    * @param w Writer for the Alloy
    * @param alloy Alloy that will be saved
    * @return the layout spec implementation for the saved alloy
    */
  def saveManifest(writer: Alloy.Writer,
      alloy: BaseAlloy[_]): BaseAlloySpecVersion = {

    val dateFormat = new java.text.SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'")
    dateFormat.setTimeZone(TimeZone.getTimeZone("UTC"))

    val manifest = JsonMethods.compact(JsonMethods.render(
      ("name" -> alloy.name) ~
        ("specVersion" -> CURRENT_SPEC.VERSION) ~
        ("idimlVersion" -> "FIXME: INCLUDE IDIML VERSION") ~
        ("createdAt" -> dateFormat.format(new Date())) ~
        ("properties" -> (
          ("java.vendor" -> System.getProperty("java.vendor")) ~
            ("java.version" -> System.getProperty("java.version")) ~
            ("os.arch" -> System.getProperty("os.arch")) ~
            ("os.name" -> System.getProperty("os.name")) ~
            ("os.version" -> System.getProperty("os.version"))
        ))
    ))
    val resource = writer.resource(MANIFEST_JSON)
    try {
      Codec.String.write(resource, manifest)
      CURRENT_SPEC
    } finally {
      resource.close
    }
  }
}

/** Interface for loading and saving Alloys according to a BaseAlloy spec
  *
  * For backwards-compatibility, the BaseAlloy loader chooses which spec
  * implementation to use based on the specVersion property in the alloy
  * manifest.
  */
private[alloy] trait BaseAlloySpecVersion {

  /** Specification version */
  val VERSION: String

  /** Loads the predictive models from the Alloy */
  def loadModels[T <: PredictResult with Buildable[T, Builder[T]]](
    engine: Engine, reader: Alloy.Reader): Map[String, PredictModel[T]]

  /** Saves all models to the Alloy */
  def saveModels[T <: PredictResult with Buildable[T, Builder[T]]](
    writer: Alloy.Writer, models: Map[String, PredictModel[T]])

  /** Saves all labels to the Alloy */
  def saveLabels(writer: Alloy.Writer, labels: Seq[Label])

  /** Loads the labels from the Alloy */
  def loadLabels(reader: Alloy.Reader): Seq[Label]

  /** Loads a JSON payload from a resource in the Alloy */
  def readJsonResource(resource: String, reader: Alloy.Reader): JValue = {
    val meta = reader.resource(resource)
    try {
      JsonMethods.parse(Codec.String.read(meta))
    } finally {
      meta.close
    }
  }

  /** Writes a JSON payload to a resource in the Alloy */
  def writeJsonResource(json: JValue, resource: String, writer: Alloy.Writer) {
    val meta = writer.resource(resource)
    try {
      Codec.String.write(meta, JsonMethods.compact(JsonMethods.render(json)))
    } finally {
      meta.close
    }
  }
}

/** Base Alloy layout spec version 1 */
private[this] object BaseAlloy_1 extends BaseAlloySpecVersion {
  val VERSION = "1"

  val MODEL_CONFIG = "models.json"
  val LABEL_CONFIG = "labels.dat"
  val MODEL_DIRECTORY = "models"

  def saveModels[T <: PredictResult with Buildable[T, Builder[T]]](
      writer: Alloy.Writer, models: Map[String, PredictModel[T]]) {
    val modelWriter = writer.within(MODEL_DIRECTORY)
    val modelEntries = models.map({ case (name, model) => {
      ArchivedModelEntry_1(name, model.getClass.getName,
        Archivable.save(model, modelWriter.within(name)))
    }})

    val modelJson = JArray(modelEntries.map(e => {
        ("name" -> e.name) ~
        ("class" -> e.`class`) ~
        ("config" -> e.config)
    }).toList)

    writeJsonResource(modelJson, MODEL_CONFIG, writer)
  }

  def saveLabels(writer: Alloy.Writer, labels: Seq[Label]) {
    val resource = new FeatureOutputStream(writer.resource(LABEL_CONFIG))
    try {
      Codec.VLuint.write(resource, labels.size)
      labels.foreach(l => l.save(resource))
    } finally {
      resource.close
    }
  }

  def loadLabels(reader: Alloy.Reader): Seq[Label] = {
    val builder = new LabelBuilder
    val resource = new FeatureInputStream(reader.resource(LABEL_CONFIG))
    try {
      val count = Codec.VLuint.read(resource)
      (0 until count).map(_ => builder.build(resource))
    } finally {
      resource.close
    }
  }

  def loadModels[T <: PredictResult with Buildable[T, Builder[T]]](
      engine: Engine, reader: Alloy.Reader): Map[String, PredictModel[T]] = {
    implicit val formats = org.json4s.DefaultFormats

    val modelReader = reader.within(MODEL_DIRECTORY)
    // load the list of models, then reify each
    readJsonResource(MODEL_CONFIG, reader)
      .extract[Seq[ArchivedModelEntry_1]]
      .map(e => {
        val klass = Class.forName(e.`class`)
        val reified = ArchiveLoader.reify[PredictModel[T]](
          klass, engine, Some(modelReader.within(e.name)), e.config)
          .getOrElse(klass.newInstance.asInstanceOf[PredictModel[T]])

        e.name -> reified
      }).toMap
  }

}

/** Top-level configuration data stored with the Alloy
  *
  * Records versioning information and associated system properties
  *
  * @param name User-friendly name for the alloy
  * @param alloySpecVersion version of BaseAlloy spec followed
  * @param idimlVersion version of idiml used to generate the Alloy
  * @param createdAt timestamp when Alloy was created
  * @param properties extra system details (JVM version, etc.)
  */
case class BaseAlloyManifest(name: String, specVersion: String,
  idimlVersion: String, createdAt: Date, properties: Map[String, String])

/** Archived top-level model in a BaseAlloy version 1.
  *
  * Used to reflectively reify the models from the Alloy.
  *
  * @param name internal name for the model
  * @param `class` model class to instantiate, must implement Archivable
  * @param config optional configuration data
  */
case class ArchivedModelEntry_1(name: String, `class`: String,
  config: Option[JObject])
