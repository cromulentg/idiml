package com.idibon.ml.feature.indexer

import com.idibon.ml.alloy.Codec
import com.idibon.ml.feature.Feature
import java.io.{DataOutputStream, DataInputStream}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import scala.collection.mutable.ArrayBuffer

/** This class represents an index of features.
  *
  * @author Michelle Casbon <michelle@idibon.com>
  */
case class Index(var values: Vector) extends Feature[Index] {

  def get = this

  // Default parameterless constructor for reflection or when you just want to load a saved Index
  def this() = this(values = Vectors.dense(0))

  /** Stores the feature to an output stream so it may be reloaded later. */
  def save(output: DataOutputStream): Unit = {
    // Convert to a SparseVector since we know that's the output of the transformer
    val sv = values.toSparse

    // Save the dimensionality of the SparseVector so we know how many times to call Codec.read() at load time
    Codec.VLuint.write(output, sv.size)

    // Save the indices. This will have length sv.size
    for (i <- sv.indices)
      Codec.VLuint.write(output, i)

    // Save the values. This will have length sv.size
    // TODO: Should this be encoded as a String instead of Int?
    for (v <- sv.values)
      Codec.VLuint.write(output, v.toInt)
  }

  /** Reloads a previously saved feature */
  def load(input: DataInputStream): Unit = {
    // Retrieve the dimensionality of the Vector so we know how many integers to expect
    val di_length = Codec.VLuint.read(input)

    // Retrieve the indices
    var di_indices = ArrayBuffer[Int]()
    1 to di_length foreach { _ => di_indices += Codec.VLuint.read(input)}

    // Retrieve the values
    var di_values = ArrayBuffer[Double]()
    1 to di_length foreach { _ => di_values += Codec.VLuint.read(input)}

    values = Vectors.sparse(di_length, di_indices.toArray, di_values.toArray)
  }
}