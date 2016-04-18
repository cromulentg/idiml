package com.idibon.ml.train.datagenerator

import java.io.{IOException, File}
import java.util.Random
import java.nio.file._

import com.typesafe.scalalogging.StrictLogging

/** Utility methods for cleaning up temporary files */
package object FileUtils extends StrictLogging {

  /** Creates a random temporary directory
    *
    * Creates a random path within the system temporary folder prefixed
    * with a caller-provided string.
    *
    * @param prefix prefix for the random temporary name
    * @param maxAttempts maximum number of random names to try before failing
    * @param rng random number generator to use for name generation
    * @return File representing the temporary directory
    */
  def createTemporaryDirectory(prefix: String, maxAttempts: Int = 10,
    rng: Random = new Random): File = {

    val systemTemp = FileSystems.getDefault.getPath(
      System.getProperty("java.io.tmpdir")).toFile

    if (!systemTemp.canWrite)
      throw new IOException("Unable to write to the system temporary folder")

    val privateDir = Stream.continually(rng.nextInt).take(maxAttempts)
      .map(id => new File(systemTemp, s"${prefix}-${id}"))
      .collectFirst({ case f: File if f.mkdir => f })

    privateDir.getOrElse({ throw new IOException("Failed to create directory") })
  }

  /** Deletes a file or directory when the JVM exits
    *
    * @param file file or directory that should be deleted
    * @return the file
    */
  def deleteAtExit(file: File): File = {
    java.lang.Runtime.getRuntime.addShutdownHook(new Thread() {
      override def run { rm_rf(file) }
    })
    file
  }

  /** Recursively deletes all files starting from any file
    *
    * Note: This method does not follow symbolic links; links will be deleted,
    * but the files will be accessible in their original locations.
    *
    * @param file Starting directory or file to delete
    */
  def rm_rf(file: File) {
    try {
      Files.walkFileTree(file.toPath, DeleteTreeVisitor)
    } catch {
      case ioe: IOException => logger.error(s"Unable to delete $file", ioe)
    }
  }
}

private[this] object DeleteTreeVisitor
    extends FileVisitor[Path] with StrictLogging {

  def postVisitDirectory(dir: Path, exc: IOException) = {
    logger.debug(s"deleting $dir")
    Files.delete(dir)
    FileVisitResult.CONTINUE
  }

  def preVisitDirectory(dir: Path, attr: attribute.BasicFileAttributes) = {
    FileVisitResult.CONTINUE
  }

  def visitFile(file: Path, attr: attribute.BasicFileAttributes) = {
    logger.debug(s"deleting $file")
    Files.delete(file)
    FileVisitResult.CONTINUE
  }

  def visitFileFailed(file: Path, exc: IOException) = {
    logger.debug(s"encountered an error at $file", exc)
    FileVisitResult.CONTINUE
  }
}
