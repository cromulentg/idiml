package com.idibon.ml.alloy

import java.io._
import java.util.jar._

import org.json4s.JsonAST.JObject
import org.json4s.native.JsonMethods.parse

import com.idibon.ml.alloy.Alloy.{Reader, Writer}
import com.idibon.ml.feature.ArchiveLoader
import com.idibon.ml.predict.PredictModel

// required for java object conversions
import collection.JavaConversions._
import scala.collection.mutable

/**
  * Created by stefankrawczyk on 1/14/16.
  */
class ScalaJarAlloy(labelModelMap: mutable.Map[String, PredictModel],
                   labelToUUID: mutable.Map[String, String]) extends BaseAlloy(labelModelMap, labelToUUID) {

  override def save(path: String): Unit = {
    val manifest: Manifest = new Manifest()
    fillManifest(manifest)

    val jos = new JarOutputStream(new FileOutputStream(new File(path)), manifest)
    // save metadata about this task
    // TODO
    // save models
    // for each model
    val baseWriter = new JarWriter("", jos)
    // JarWriter writer = baseWriter.namespace("label");
    // model.save(writer)
    jos.close
  }

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
    attr.put(Attributes.Name.IMPLEMENTATION_VERSION, "0.0.1")
  }
}

object ScalaJarAlloy {

  def load(path: String): ScalaJarAlloy = {
    val jarFile: File = new File(path)
    val jar: JarFile = new JarFile(jarFile)
    // read top level models & get classes
    val reader: JarReader = new JarReader("", jar)
    // using reflection create that class and call the load method and reify models.
    /* Something like this
    ArchiveLoader.reify[PredictModel](
      Class.forName("com.idibon.ml.predict.rules.DocumentRules"), reader,
      Some(parse("{\"label\": \"aLabel\"}").asInstanceOf[JObject]))
    */
    val labelModels: mutable.Map[String, PredictModel] = new mutable.HashMap[String, PredictModel]
    val labelToUUID: mutable.Map[String, String] = new mutable.HashMap[String, String]
    // instantiate other objects
    jar.close()
    // return fresh instance
    return new ScalaJarAlloy(labelModels, labelToUUID)
  }
}

private class JarReader(currentPath: String, jarFile: JarFile) extends Alloy.Reader {

  override def within(namespace: String): Reader = {
    val newNamespace = s"${currentPath}${namespace}/"
    new JarReader(newNamespace, jarFile)
  }

  override def resource(resourceName: String): DataInputStream = {
    val je = new JarEntry(s"${currentPath}${resourceName}")
    new DataInputStream(jarFile.getInputStream(je))
  }
}


private class JarWriter(currentPath: String, jos: JarOutputStream) extends Alloy.Writer {

  override def within(namespace: String): Writer = {
    val newNamespace = s"${currentPath}${namespace}/"
    val je = new JarEntry(newNamespace)
    jos.synchronized{
      jos.putNextEntry(je)
      jos.closeEntry()
    }
    new JarWriter(newNamespace, jos)
  }

  override def resource(resourceName: String): DataOutputStream = {
    val je = new JarEntry(s"${currentPath}${resourceName}") // "/" is taken care of.
    new IdibonJarDataOutputStream(new ByteArrayOutputStream(), jos, je)
  }
}


private class IdibonJarDataOutputStream(baos: ByteArrayOutputStream,
                                        jos: JarOutputStream,
                                        je: JarEntry) extends DataOutputStream(baos) {

  override def close(): Unit = {
    je.setTime(System.currentTimeMillis())
    jos.synchronized{
      jos.putNextEntry(je)
      jos.write(baos.toByteArray())
      jos.closeEntry()
    }
    baos.close()
  }
}
