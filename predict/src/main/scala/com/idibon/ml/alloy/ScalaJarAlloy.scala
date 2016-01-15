package com.idibon.ml.alloy

import java.io._
import java.util.jar._

import com.typesafe.scalalogging.StrictLogging
import org.json4s.JsonAST._
import org.json4s.native.JsonMethods.{parse, compact, render}
import org.json4s.JsonDSL._

import com.idibon.ml.alloy.Codec
import com.idibon.ml.alloy.Alloy.{Reader, Writer}
import com.idibon.ml.feature.{Archivable, ArchiveLoader}
import com.idibon.ml.predict.PredictModel

// required for java object conversions
import collection.JavaConversions._
import scala.collection.mutable

/**
  * Rough draft of a Jar backed Alloy.
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
class ScalaJarAlloy(labelModelMap: mutable.Map[String, PredictModel],
                    labelToUUID: mutable.Map[String, String])
  extends BaseAlloy(labelModelMap, labelToUUID) with StrictLogging {


  /**
    * Saves the Alloy as a Jar file to the specified path.
    * @param path
    */
  override def save(path: String): Unit = {
    logger.info(s"Attemping to save Alloy [v. ${ScalaJarAlloy.CURRENT_VERSION}] to ${path}.")
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
    saveMapOfData(
      baseWriter,
      labelToUUID.map(x => (x._1, JString(x._2))).toList,
      ScalaJarAlloy.LABEL_UUID)
    // save class types of models
    saveMapOfData(
      baseWriter,
      labelModelMap.map(x => (x._1, JString(x._2.getType()))).toList,
      ScalaJarAlloy.MODEL_CLASS)
    // save models
    saveMapOfData(
      baseWriter,
      labelModelMap
        .par.map(
          // for each model save it and get the JObject back
          x =>
            (x._1, Archivable.save(x._2, baseWriter.within(x._1)).getOrElse(JNothing))
        ).toList,
      ScalaJarAlloy.MODEL_META)
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
    attr.put(Attributes.Name.IMPLEMENTATION_VERSION, ScalaJarAlloy.CURRENT_VERSION)
  }
}

object ScalaJarAlloy extends StrictLogging {

  val CURRENT_VERSION: String = "0.0.1"

  val LABEL_UUID: String = "labels-uuid.json"

  val MODEL_CLASS: String = "model-class.json"

  val MODEL_META: String = "model-meta.json"

  /**
    * Static method to load an alloy from a Jar file.
    *
    * @param path
    * @return
    */
  def load(path: String): ScalaJarAlloy = {
    implicit val formats = org.json4s.DefaultFormats
    val jarFile: File = new File(path)
    val jar: JarFile = new JarFile(jarFile)
    // manifest
    val manifest: Manifest = jar.getManifest()
    // get the version out & check it.
    val version = manifest.getMainAttributes().getValue(Attributes.Name.IMPLEMENTATION_VERSION)
    version match {
      case "0.0.1" => logger.info(s"Attemping to load version [v. ${version}].")
      case _ => throw new IOException(s"Unable to load, unhandled version ${version}")
    }
    // base reader
    val baseReader: JarReader = new JarReader("", jar)
    // get labels, classes and model metadata
    val labelToUUIDMap = readMapOfData(baseReader, LABEL_UUID).extract[Map[String, String]]
    val modelClassesMap = readMapOfData(baseReader, MODEL_CLASS).extract[Map[String, String]]
    val modelMetadata = readMapOfData(baseReader, MODEL_META)
    // using reflection create that class and call the load method and reify models.
    val labelModels: mutable.Map[String, PredictModel] = new mutable.HashMap[String, PredictModel]
    val labelToUUID: mutable.Map[String, String] = new mutable.HashMap[String, String]
    for((label, modelClass) <- modelClassesMap) {
      // Reify the model.
      val model: PredictModel = ArchiveLoader.reify[PredictModel](
        Class.forName(modelClass),
        baseReader.within(label),
        // extra the right model metadata to send down
        Some((modelMetadata \ label).extract[JObject])).get
      // have to create these maps this way because we're dealing with Java in the end.
      labelModels.put(label, model)
      labelToUUID.put(label, labelToUUIDMap.getOrElse(label, ""))
    }
    // instantiate other objects
    jar.close()
    // return fresh instance
    return new ScalaJarAlloy(labelModels, labelToUUID)
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
    implicit val formats = org.json4s.DefaultFormats
    val metaInputStream = baseReader.resource(resourceName)
    val rawJSON = Codec.String.read(metaInputStream)
    metaInputStream.close()
    return parse(rawJSON).asInstanceOf[JObject]
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
