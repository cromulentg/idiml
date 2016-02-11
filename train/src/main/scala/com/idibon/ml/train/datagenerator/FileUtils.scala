package com.idibon.ml.train.datagenerator

import java.io.{IOException, File}
import java.nio.file._
import com.typesafe.scalalogging.StrictLogging

/** Utility methods for cleaning up temporary files */
package object FileUtils extends StrictLogging {

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
