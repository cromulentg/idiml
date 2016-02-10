package com.idibon.ml.alloy

import com.idibon.ml.predict._
import com.idibon.ml.feature._

import org.json4s._
import org.json4s.native.JsonMethods

/** Optional Alloy trait for Furnaces to include validation examples
  */
trait HasValidationData {
  /** List of JSON documents to use for validation */
  def validationExamples: Seq[JObject]
}

/** Optional Alloy trait for storing training configuration data */
trait HasTrainingConfig {
  /** JSON object containing the training configuration */
  def trainingConfig: JObject
}

/** Companion object for HasValidationData */
object HasValidationData {

  val VALIDATION_RESOURCE = "validation.dat"

  /** Loads the validation examples from the alloy and checks for correctness
    *
    * @throw ValidationException if the predicted results are invalid
    */
  def validate[T <: PredictResult with Buildable[T, Builder[T]]](
      reader: Alloy.Reader, alloy: BaseAlloy[T]) {
    val resource = new FeatureInputStream(reader.resource(VALIDATION_RESOURCE))
    if (resource == null) return;

    try {
      val exampleCount = Codec.VLuint.read(resource)
      val modelCount = Codec.VLuint.read(resource)
      val modelNames = (0 until modelCount).map(_ => Codec.String.read(resource))
        (0 until exampleCount).foreach(_ => {
          val docJson = JsonMethods.parse(Codec.String.read(resource))
          val doc = Document.document(docJson.asInstanceOf[JObject])
          modelNames.foreach(n => {
            val expectCount = Codec.VLuint.read(resource)
            val expected = (0 until expectCount).map(_ => {
              resource.readBuildable.asInstanceOf[T]
            })
            val results = alloy.models(n).predict(doc, PredictOptions.DEFAULT)
            if (results.size != expected.size)
              throw new ValidationError(s"${alloy.name} ${n}")
            results.zip(expected).foreach({ case (actual, gold) => {
              if (!actual.isCloseEnough(gold))
                throw new ValidationError(s"${alloy.name} ${n}")
            }})
          })
        })
    } finally {
      resource.close
    }
  }

  /** Saves an alloy with validation data to a persistent device
    *
    * @param writer alloy writer
    * @param alloy the alloy (with validation data) being saved
    */
  def save[T <: PredictResult with Buildable[T, Builder[T]]](
      writer: Alloy.Writer, alloy: BaseAlloy[T] with HasValidationData) {

    val resource = new FeatureOutputStream(writer.resource(VALIDATION_RESOURCE))
    val modelNames = alloy.models.map(_._1)

    try {
      Codec.VLuint.write(resource, alloy.validationExamples.size)
      Codec.VLuint.write(resource, modelNames.size)
      modelNames.foreach(n => Codec.String.write(resource, n))

      alloy.validationExamples.foreach(doc => {
        Codec.String.write(resource, JsonMethods.compact(JsonMethods.render(doc)))
        val document = Document.document(doc)
        modelNames.foreach(n => {
          val results = alloy.models(n).predict(document, PredictOptions.DEFAULT)
          Codec.VLuint.write(resource, results.size)
          results.foreach(p => resource.writeBuildable(p))
        })
      })
    } finally {
      resource.close
    }
  }
}

/** Companion object for HasTrainingConfig */
object HasTrainingConfig {
  val CONFIG_RESOURCE = "training.json"

  def save(writer: Alloy.Writer, alloy: HasTrainingConfig) {
    val config = JsonMethods.compact(JsonMethods.render(alloy.trainingConfig))
    val resource = writer.resource(CONFIG_RESOURCE)
    try {
      Codec.String.write(resource, config)
    } finally {
      resource.close
    }
  }
}
