package com.idibon.ml.alloy

import java.io._
import scala.collection.mutable.HashMap

/** In-memory I/O for alloys */
class MemoryAlloyReader(data: Map[String, Array[Byte]]) extends Alloy.Reader {

  def within(namespace: String): Alloy.Reader = new SubReader(namespace + "/")

  def resource(name: String): DataInputStream = {
    data.get(name)
      .map(bytes => new DataInputStream(new ByteArrayInputStream(bytes)))
      .getOrElse(null)
  }

  private class SubReader(path: String) extends Alloy.Reader {
    def within(namespace: String): Alloy.Reader = new SubReader(path + namespace + "/")

    def resource(name: String): DataInputStream = {
      data.get(path + name)
        .map(bytes => new DataInputStream(new ByteArrayInputStream(bytes)))
        .getOrElse(null)
    }
  }
}


class MemoryAlloyWriter(data: HashMap[String, Array[Byte]]) extends Alloy.Writer {
  def within(namespace: String): Alloy.Writer = new SubWriter(namespace + "/")

  def resource(name: String): DataOutputStream = new Stage(name)

  private class SubWriter(path: String) extends Alloy.Writer {
    def within(namespace: String): Alloy.Writer = new SubWriter(path + namespace + "/")

    def resource(name: String): DataOutputStream = new Stage(path + name)
  }

  private class Stage(key: String,
    bytes: ByteArrayOutputStream = new ByteArrayOutputStream)
      extends DataOutputStream(bytes) {
    override def close {
      super.close
      data.synchronized { data += (key -> bytes.toByteArray) }
    }
  }
}
