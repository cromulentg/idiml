package com.idibon.ml.feature

import java.io.{DataInputStream, DataOutputStream}

/** Mixin trait for features that support saving and re-loading
  *
  * Implementations are paired to Builder classes that read data from an
  * InputStream and instantiate the object, so that objects may be
  * immutable.
  */
trait Buildable[T <: Buildable[T, Builder[T]], +U <: Builder[T]] {

  /** Stores the data to an output stream so it may be reloaded later.
    *
    * @param output - Output stream to write data
    */
  def save(output: DataOutputStream)
}

trait Builder[T] {
  /** Instantiates and loads an object from an input stream
    *
    * @param input - Data stream where object was previously saved
    */
  def build(input: DataInputStream): T
}
