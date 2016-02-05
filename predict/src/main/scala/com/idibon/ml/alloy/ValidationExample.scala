package com.idibon.ml.alloy

import org.json4s.JObject
import org.json4s.native.Serialization.writePretty
import org.json4s.native.JsonMethods.parse

import com.idibon.ml.feature.{FeatureInputStream, FeatureOutputStream, Builder, Buildable}
import com.idibon.ml.predict.{PredictResult, ClassificationBuilder, Classification, Document}

/**
  * Class to encapsulate validation examples.
  */
case class ValidationExample[T <: PredictResult with Buildable[T, Builder[T]]](document: Document, predictions: Seq[T])
  extends Buildable[ValidationExample[T], ValidationExampleBuilder[T]]{
  /** Stores the data to an output stream so it may be reloaded later.
    *
    * @param output - Output stream to write data
    */
  override def save(output: FeatureOutputStream): Unit = {
    implicit val formats = org.json4s.DefaultFormats
    Codec.String.write(output, writePretty(document.json))
    Codec.VLuint.write(output, predictions.size)
    predictions.foreach(pred => pred.save(output))
  }
}

/**
  * Builder to bring them back to life.
  */
class ValidationExampleBuilder[T <: PredictResult with Buildable[T, Builder[T]]](builder: Builder[T]) extends Builder[ValidationExample[T]] {
  /** Instantiates and loads an object from an input stream
    *
    * @param input - Data stream where object was previously saved
    */
  override def build(input: FeatureInputStream): ValidationExample[T] = {
    implicit val formats = org.json4s.DefaultFormats
    val jsonDoc = parse(Codec.String.read(input)).extract[JObject]
    val predictionSize = Codec.VLuint.read(input)
    val predictions = (0 until predictionSize).map(_ => {
      builder.build(input)
    })
    new ValidationExample(Document.document(jsonDoc), predictions)
  }
}

case class ValidationExamples[T <: PredictResult with Buildable[T, Builder[T]]](examples: List[ValidationExample[T]])
  extends Buildable[ValidationExamples[T], ValidationExamplesBuilder[T]] {
  /** Stores the data to an output stream so it may be reloaded later.
    *
    * @param output - Output stream to write data
    */
  override def save(output: FeatureOutputStream): Unit = {
    Codec.VLuint.write(output, examples.size)
    examples.foreach(example => example.save(output))
  }
}

class ValidationExamplesBuilder[T <: PredictResult with Buildable[T, Builder[T]]](builder: Builder[T]) extends Builder[ValidationExamples[T]] {
  /** Instantiates and loads an object from an input stream
    *
    * @param input - Data stream where object was previously saved
    */
  override def build(input: FeatureInputStream): ValidationExamples[T] = {
    val exampleSize = Codec.VLuint.read(input)
    val exampleBuilder = new ValidationExampleBuilder[T](builder)
    val examples = (0 until exampleSize).map(_ => {
      exampleBuilder.build(input)
    })
    new ValidationExamples(examples.toList)
  }
}

/**
  * Exception to throw with details when the model doesn't validate against itself.
  * @param msg
  */
class ValidationError(msg: String) extends Exception(msg)
