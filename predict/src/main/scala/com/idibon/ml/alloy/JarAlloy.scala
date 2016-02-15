package com.idibon.ml.alloy

import java.io._
import java.util.jar._
import com.idibon.ml.common.Engine
import com.idibon.ml.predict.PredictResult
import com.idibon.ml.feature.{Buildable, Builder}
import com.idibon.ml.predict.ml.TrainingSummary

/**
  * Use this to save & load Jar backed Alloys.
  *
  * @author "G.K. <gary@idibon.com>"
  */
object JarAlloy {

  /**
    * Loads a BaseAlloy stored in a JAR, optionally validating the models
    *
    * @param engine containts the spark context.
    * @param file the file to load from.
    * @param validate whether to validate against the internal examples.
    * @tparam T The type of PredictResult we're expecting the alloy to produce.
    * @return a JarAlloy.
    */
  def load[T <: PredictResult with Buildable[T, Builder[T]]](
      engine: Engine, file: File, validate: Boolean): Alloy[T] = {
    val jar = new JarFile(file)
    try {
      val reader = new JarAlloyReader(jar)
      val alloy = BaseAlloy.load[T](engine, reader)
      if (validate) HasValidationData.validate(reader, alloy)
      alloy
    } finally {
      jar.close
    }
  }

  /**
    * Gets training summaries from the alloy if they exist.
    * @param file location of the jar alloy file.
    * @return
    */
  def getTrainingSummaries(file: File): List[TrainingSummary] = {
    val jar = new JarFile(file)
    try {
      val reader = new JarAlloyReader(jar)
      HasTrainingSummary.get(reader).toList
    } finally {
      jar.close()
    }
  }

  /**
    * Saves a BaseAlloy to a JAR
    *
    * @param alloy JarAlloy to save.
    * @param file file to save alloy to.
    * @tparam T the PredictResult type this alloy uses.
    */
  def save[T <: PredictResult with Buildable[T, Builder[T]]](
      alloy: Alloy[T], file: File) = {
    val jos = new JarOutputStream(new FileOutputStream(file))
    try {
      val writer = new JarAlloyWriter(jos)
      alloy.save(writer)
    } finally {
      jos.close
    }
  }
}

/** Reads alloy data from a JAR file.
  *
  * @param jarFile
  */
case class JarAlloyReader(jarFile: JarFile) extends Alloy.Reader {

  def within(namespace: String): Alloy.Reader = new SubReader(namespace + "/")

  def resource(name: String): DataInputStream = {
    val entry = jarFile.getJarEntry(name)
    if (entry != null) new DataInputStream(jarFile.getInputStream(entry)) else null
  }

  /** Inner class used for reading internal namespaces */
  private class SubReader(path: String) extends Alloy.Reader {
    def within(namespace: String): Alloy.Reader = new SubReader(path + namespace + "/")

    def resource(name: String): DataInputStream = {
      val entry = jarFile.getJarEntry(path + name)
      if (entry != null) new DataInputStream(jarFile.getInputStream(entry)) else null
    }
  }
}

/** Writes alloy data to a JAR file.
  *
  * @param jarStream
  */
case class JarAlloyWriter(jarStream: JarOutputStream) extends Alloy.Writer {

  def within(namespace: String): Alloy.Writer = new SubWriter(namespace + "/")

  def resource(name: String): DataOutputStream = new StageStream(new JarEntry(name))

  private class SubWriter(path: String) extends Alloy.Writer {
    def within(namespace: String): Alloy.Writer = new SubWriter(path + namespace + "/")

    def resource(name: String): DataOutputStream = new StageStream(new JarEntry(path + name))
  }

  /** Serializes writes to the JAR file
    *
    * JAR files entries must be written sequentially, and in entirety,
    * before advancing to the next entry. Alloy.Writer has no provisions
    * for this limitation, so resource writes must be staged so that
    * entire resources are written to the alloy atomically
    */
  private class StageStream(entry: JarEntry,
    stage: ByteArrayOutputStream = new ByteArrayOutputStream)
      extends DataOutputStream(stage) {

    override def close {
      super.close
      JarAlloyWriter.this.synchronized {
        jarStream.putNextEntry(entry)
        jarStream.write(stage.toByteArray)
        jarStream.closeEntry
      }
    }
  }
}
