package com.idibon.ml.alloy

import java.io._
import java.util.jar._

import com.idibon.ml.feature.{Builder, Buildable, FeatureInputStream, FeatureOutputStream}
import com.typesafe.scalalogging.StrictLogging
import org.json4s.JsonAST._
import org.json4s.native.JsonMethods.{parse, render, compact}
import com.idibon.ml.common.Engine
import org.json4s.JsonDSL._

import com.idibon.ml.alloy.Alloy.{Reader, Writer}
import com.idibon.ml.common.{Archivable, ArchiveLoader}
import com.idibon.ml.predict._

// required for java object conversions
import scala.collection.JavaConverters._
import scala.collection.mutable

/**
  * Canonical Alloy implementation backed by a JAR file.
  * <p>
  * This is very simple, and allows you to write to a jar as if it was a file system.
  * Writing:
  * It doesn't do any signing or anything complex with Manifests.
  * Reading:
  * It doesn't do anything interesting, other than returning streams for reading.
  * <p>

  * http://docs.oracle.com/javase/8/docs/technotes/guides/jar/jar.html - jar spec
  * https://docs.oracle.com/javase/8/docs/api/java/util/jar/package-summary.html - code
  *
  * @author "Stefan Krawczyk <stefan@idibon.com>"
  */
class JarAlloy[T <: PredictResult with Buildable[T, Builder[T]]](models: Map[String, PredictModel[T]],
                                                                 uuids: Map[String, Label],
                                                                 validationExamples: Map[String, ValidationExamples[T]] = Map[String, ValidationExamples[T]]())
  extends BaseAlloy[T](models.values.toList.asJava, uuids.asJava, validationExamples.asJava)
      with StrictLogging {

  /**
    * This method validates the results at training vs now.
    *
    * throws ValidationError if any results are not within our tolerances.
    */
  override def validate(): Unit = {
    val errors = JarAlloy.validate[T](models, validationExamples)
    if (!errors.isEmpty){
      logger.error(errors)
      throw new ValidationError(errors)
    }
  }

  /**
    * Saves the Alloy as a Jar file to the specified path.
    *
    * @param path
    */
  override def save(path: String): Unit = {
    logger.info(s"Attemping to save Alloy [v. ${JarAlloy.CURRENT_VERSION}] to ${path}.")
    // check that we can save to the path
    val file: File = new File(path)
    ensureBaseDirectoryExists(file)
    // create manifest for information about the jar and this current state of code.
    val manifest: Manifest = new Manifest()
    fillManifest(manifest)
    // create stream to write to
    val jos = new JarOutputStream(new FileOutputStream(file), manifest)
    // create base writer that will be used to write concurrently to.
    val baseWriter = new JarWriter("", jos)
    // save labels map
    saveUUIDsToLabel(
      baseWriter,
      JarAlloy.LABEL_UUID)
    // save class types of models
    saveMapOfData(
      baseWriter,
      this.models.map({ case (label, m) => (label, JString(m.getClass.getName))}).toList,
      JarAlloy.MODEL_CLASS)
    // save validation data
    saveValidationResults(baseWriter)
    // save models
    saveMapOfData(
      baseWriter,
      this.models.par.map({ case (label, model) => {
          // for each model save it and get the JObject back
          (label, Archivable.save(model, baseWriter.within(label)).getOrElse(JNothing))
        }}).toList,
      JarAlloy.MODEL_META)
    // TODO: save more schtuff about this task
    jos.close()
  }

  /**
    * Helper method to ensure the path we're saving to has
    * all the directories created for it up to that point.
    *
    * @param file
    */
  def ensureBaseDirectoryExists(file: File): Unit = {
    val parentPath = file.getParent()
    if (parentPath != null) new File(parentPath).mkdirs()
  }

  /**
    * Helper method to take a map represented as a List of JFields and save
    * it to a particular resource at the position in the writer.
    *
    * @param writer
    * @param mapOfData
    * @param resourceName
    */
  def saveMapOfData(writer: JarWriter, mapOfData: List[JField], resourceName: String): Unit = {
    val metaOutputStream = writer.resource(resourceName)
    Codec.String.write(metaOutputStream, compact(render(JObject(mapOfData))))
    metaOutputStream.close()
  }

  /**
    * Helper method to take a map represented as a List of JFields and save
    * it to a particular resource at the position in the writer.
    *
    * @param writer
    * @param resourceName
    */
  def saveUUIDsToLabel(writer: JarWriter, resourceName: String): Unit = {
    val metaOutputStream = new FeatureOutputStream(writer.resource(resourceName))
    Codec.VLuint.write(metaOutputStream, uuids.size)
    uuids.foreach({ case (uuid, label) => label.save(metaOutputStream) })
    metaOutputStream.close()
  }

  /**
    * Helper method to save Validation example results.
    *
    * @param writer
    */
  def saveValidationResults(writer: JarWriter): Unit = {
    val validationWriter = writer.within(JarAlloy.VALIDATION_LOCATION)
    val safeguard = validationWriter.resource(JarAlloy.VALIDATION_LOCATION_SAFEGUARD)
    safeguard.writeBoolean(validationExamples.isEmpty)
    safeguard.close()
    validationExamples.foreach { case (modelName, examples) => {
      val validationOutputStream = new FeatureOutputStream(validationWriter.resource(modelName))
      examples.save(validationOutputStream)
      validationOutputStream.close()
    }}
  }

  /**
    * Helper method to fill a manifest.
    *
    * The contents of this currently are just placeholders.
    *
    * @param manifest
    */
  private def fillManifest(manifest: Manifest) {
    val attr: Attributes = manifest.getMainAttributes
    attr.put(Attributes.Name.MANIFEST_VERSION, "0.0.1")
    attr.put(new Attributes.Name("Created-By"), "Idibon Inc.")
    attr.put(new Attributes.Name("JVM-Version"), System.getProperty("java.version"))
    attr.put(Attributes.Name.SPECIFICATION_TITLE, "Idibon IdiJar-Alloy for Prediction")
    attr.put(Attributes.Name.SPECIFICATION_VENDOR, "Idibon Inc.")
    attr.put(Attributes.Name.SPECIFICATION_VERSION, "0.0.1")
    attr.put(Attributes.Name.IMPLEMENTATION_TITLE, "com.idibon.ml")
    attr.put(Attributes.Name.IMPLEMENTATION_VENDOR, "Idibon Inc.")
    // save version number -- use this for determining what versioning of loading/saving jars.
    attr.put(Attributes.Name.IMPLEMENTATION_VERSION, JarAlloy.CURRENT_VERSION)
  }
}

object JarAlloy extends StrictLogging {

  // the implementation version of this scala jar alloy.
  val CURRENT_VERSION: String = "0.0.1"

  val LABEL_UUID: String = "labels-uuid.json"

  val MODEL_CLASS: String = "model-class.json"

  val MODEL_META: String = "model-meta.json"

  val VALIDATION_LOCATION: String = "validationExamples"

  val VALIDATION_LOCATION_SAFEGUARD: String = "number"


  /**
    * Static method to load an alloy from a Jar file and validate it against the internally stored
    * results.
    *
    * @param engine the engine that contains the spark context.
    * @param path place where the Jar files lives.
    * @param validationBuilder Builder that knows how to reload validation examples.
    * @tparam T The type of predictResult we expect to create and make
    * @return an alloy ready for combat.
    */
  def loadAndValidate[T <: PredictResult with Buildable[T, Builder[T]]](engine: Engine,
                                                                        path: String,
                                                                        validationBuilder: ValidationExamplesBuilder[T]): JarAlloy[T] = {
    val jarFile: File = new File(path)
    val jar: JarFile = new JarFile(jarFile)
    checkVersion(jar)
    // base reader
    val baseReader: JarReader = new JarReader("", jar)
    val uuidToLabel = readUUIDsToLabel(baseReader, LABEL_UUID)
    val labelModels = getModels(engine, baseReader)
    // instantiate other objects
    val examples = loadValidationResults[T](baseReader, labelModels.keys.toList, validationBuilder)
    jar.close()
    // return fresh instance
    val alloy =  new JarAlloy(labelModels, uuidToLabel, examples)
    alloy.validate()
    return alloy
  }

  /**
    * Static method to load an alloy from a Jar file.
    *
    * @param engine the engine that contains the spark context.
    * @param path place where the Jar files lives.
    * @tparam T The type of predictResult we expect to create and make
    * @return an alloy ready for combat.
    */
  def load[T <: PredictResult with Buildable[T, Builder[T]]](engine: Engine,
                                                             path: String): JarAlloy[T] = {
    val jarFile: File = new File(path)
    val jar: JarFile = new JarFile(jarFile)
    checkVersion(jar)
    // base reader
    val baseReader: JarReader = new JarReader("", jar)
    // get labels, classes and model metadata
    val uuidToLabel = readUUIDsToLabel(baseReader, LABEL_UUID)
    val labelModels = getModels(engine, baseReader)
    jar.close()
    // return fresh instance
    return new JarAlloy(labelModels, uuidToLabel)
  }

  /**
    * Checks the version in the Jar is one that we support.
    * @param jar
    * @tparam T
    */
  def checkVersion[T <: PredictResult with Buildable[T, Builder[T]]](jar: JarFile): Unit = {
    // manifest
    val manifest: Manifest = jar.getManifest()
    // get the version out & check it.
    val version = manifest.getMainAttributes().getValue(Attributes.Name.IMPLEMENTATION_VERSION)
    version match {
      case "0.0.1" => logger.info(s"Attemping to load version [v. ${version}].")
      case _ => throw new IOException(s"Unable to load, unhandled version ${version}")
    }
  }

  /**
    * Private method to loads the models and labels from the JAR and pass them back.
    * @param engine
    * @param baseReader
    * @tparam T
    * @return
    */
  private def getModels[T <: PredictResult with Buildable[T, Builder[T]]](engine: Engine,
                                                                          baseReader: JarReader):
  Map[String, PredictModel[T]] = {
    implicit val formats = org.json4s.DefaultFormats
    val modelClassesMap = readMapOfData(baseReader, MODEL_CLASS).extract[Map[String, String]]
    val modelMetadata = readMapOfData(baseReader, MODEL_META)
    // using reflection create that class and call the load method and reify models.
    val labelModels = new mutable.HashMap[String, PredictModel[T]]
    for ((label, modelClass) <- modelClassesMap) {
      // Reify the model.
      val model = ArchiveLoader.reify[PredictModel[T]](
        Class.forName(modelClass), engine, Some(baseReader.within(label)),
        // extra the right model metadata to send down
        Some((modelMetadata \ label).extract[JObject])).get
      // have to create these maps this way because we're dealing with Java in the end.
      labelModels.put(label, model)
    }
    labelModels.toMap
  }

  /**
    * Helper method to read in a map of data. It returns it as a JObject since
    * the values in this map could be anything.
    *
    * @param baseReader
    * @param resourceName
    * @return
    */
  private def readMapOfData(baseReader: JarReader, resourceName: String): JObject = {
    val metaInputStream = baseReader.resource(resourceName)
    val rawJSON = Codec.String.read(metaInputStream)
    metaInputStream.close()
    return parse(rawJSON).asInstanceOf[JObject]
  }

  /**
    * Helper method to read in a UUIDs to Label mapping.
    *
    * @param baseReader
    * @param resourceName
    * @return
    */
  private def readUUIDsToLabel(baseReader: JarReader, resourceName: String): Map[String, Label] = {
    val metaInputStream = new FeatureInputStream(baseReader.resource(resourceName))
    val builder = new LabelBuilder()
    val size = Codec.VLuint.read(metaInputStream)
    val labels = (0 until size).map(_ => {
      val label = builder.build(metaInputStream)
      (label.uuid.toString, label)
    })
    metaInputStream.close()
    return labels.toMap
  }

  /**
    * Helper method to load validation examples
    *
    * @param baseReader
    * @param modelNames
    * @param builder
    * @tparam T
    * @return
    */
  def loadValidationResults[T <: PredictResult with Buildable[T, Builder[T]]](baseReader: JarReader,
                                                                              modelNames: List[String],
                                                                              builder: ValidationExamplesBuilder[T]): Map[String, ValidationExamples[T]] = {
    val validationReader = baseReader.within(JarAlloy.VALIDATION_LOCATION)
    val safeguard = validationReader.resource(JarAlloy.VALIDATION_LOCATION_SAFEGUARD)
    val hasNoValidationExamples = safeguard.readBoolean()
    safeguard.close()
    if (hasNoValidationExamples) return Map()
    modelNames.map(modelName => {
      val validationInputStream = new FeatureInputStream(validationReader.resource(modelName))
      val examples = builder.build(validationInputStream)
      validationInputStream.close()
      (modelName, examples)
    }).toMap
  }

  /**
    * Static helper method to validate a set of examples with a model's predictions.
    * @param models
    * @param validationExamples
    * @tparam T
    * @return
    */
  def validate[T <: PredictResult with Buildable[T, Builder[T]]](models: Map[String, PredictModel[T]],
                                                                 validationExamples: Map[String, ValidationExamples[T]]): String = {
    val options = PredictOptions.DEFAULT
    validationExamples.flatMap{case(modelName, examples) => {
      examples.examples.map(example => {
        (models(modelName).predict(example.document, options), example.predictions)
      })// flatten so all predictions are in a single list of (newPrediction, oldPrediction)
        .map(x => x.zipped).flatten
        // remove predictions that are good
        .filter({case (newPrediction, oldPrediction) => !newPrediction.isCloseEnough(oldPrediction)})
        // create messages for bad predictions
        .map { case (newPrediction, oldPrediction) => {
        s"Failed to match prediction for new:\n${newPrediction}\nvs old:\n${oldPrediction}\n"
        }
      }
    }}.mkString("\n")
  }
}

/**
  * Returns a JarReader set at a certain path in the JarFile.
  *
  * @param currentPath
  * @param jarFile
  */
private class JarReader(currentPath: String, jarFile: JarFile) extends Alloy.Reader {

  /**
    * Returns a reader that will read resources from that path.
    *
    * @param namespace
    * @return
    */
  override def within(namespace: String): Reader = {
    val newNamespace = s"$currentPath$namespace/"
    new JarReader(newNamespace, jarFile)
  }

  /**
    * Returns a stream to be able to read from for that resource.
    *
    * @param resourceName
    * @return
    */
  override def resource(resourceName: String): DataInputStream = {
    val je = new JarEntry(s"$currentPath$resourceName")
    new DataInputStream(jarFile.getInputStream(je))
  }
}

/**
  * Constructor that sets up from what path a Writer will start writing at.
  *
  * @param currentPath
  * @param jos
  */
private class JarWriter(currentPath: String, jos: JarOutputStream) extends Alloy.Writer {

  /**
    * Returns a writer that will ensure that the "namespace" exists and is ready for use.
    *
    * Essentially it just creates a directory entry and returns a new JarWriter to write
    * from that directory as its base.
    *
    * @param namespace
    * @return
    */
  override def within(namespace: String): Writer = {
    val newNamespace = s"$currentPath$namespace/"
    val je = new JarEntry(newNamespace)
    // ensure only one thread is writing to the JarOutputStream at any time.
    jos.synchronized {
      jos.putNextEntry(je)
      jos.closeEntry()
    }
    new JarWriter(newNamespace, jos)
  }

  override def resource(resourceName: String): DataOutputStream = {
    val je = new JarEntry(s"$currentPath$resourceName") // "/" is taken care of.
    new IdibonJarDataOutputStream(new ByteArrayOutputStream(), jos, je)
  }
}

/**
  * Wrapper class to allow "concurrent" writing to the Jar being
  * written to.
  *
  * The premise is that we do a bait and switch. We pretend to give
  * them an object that writes to the JAR but in fact, it's just
  * a ByteArrayOutputStream. This allows multiple threads to write
  * in parallel. Then, as good citizens, they try to close the stream,
  * we then actually take what they've written and shove it under the
  * specified JarEntry in the Jar, accounting for the fact that only
  * one thread can write at any one time by "synchronizing" access to
  * the JarOutputStream.
  *
  * @author "Stefan Krawczyk <stefan@idibon.com>"
  * @param baos
  * @param jos
  * @param je
  */
private class IdibonJarDataOutputStream(baos: ByteArrayOutputStream,
                                        jos: JarOutputStream,
                                        je: JarEntry) extends DataOutputStream(baos) {
  /**
    * Override the close method to actually write to the JAR file.
    *
    * @throws IOException
    */
  override def close(): Unit = {
    je.setTime(System.currentTimeMillis())
    // we only every want one of these instances to write to the single output stream that
    // writes to the JAR. I believe that is all I need to do?
    jos.synchronized {
      jos.putNextEntry(je)
      jos.write(baos.toByteArray())
      jos.closeEntry()
    }
    baos.close()
  }
}
